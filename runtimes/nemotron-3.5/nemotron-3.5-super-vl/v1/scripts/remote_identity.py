#!/usr/bin/env python3
"""Read-only identity primitives. Hashes are point-in-time, not a filesystem lease.

Trust root: the kernel, system Python/NSS, root and the current UID. Group write
requires proof of the current owner's private primary LOCAL group, not a matching
name/GID alone. Effective POSIX ACL grants are checked through held descriptors.
Only known local POSIX filesystems are supported; unknown authority/access
mechanisms fail closed. Root-owned sticky ancestors retain their narrow exception
(never the admitted root). Read-only model payloads may have stable hardlinks;
binaries, dependencies and writable/download destinations remain single-link.
The operator must keep admitted trees and account authority unchanged while
serving. This is point-in-time admission, NOT a lease or hostile root/same-UID
isolation (including transient change-and-restore, preexisting writer FDs or
credentials retained after group removal). Nothing is chmod/chown'd or repaired.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import ctypes
import errno
import grp
import struct
import hashlib
import base64
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import stat
import subprocess
import sys

# Upstream documentation is not consumed by the native loader. No glob exemptions.
MODEL_DOCUMENTS = frozenset({'README.md', 'LICENSE', '.gitattributes'})
# `hf download --local-dir` leaves resumable-download bookkeeping here, and the managed
# downloader (fetch_models.inspect_payload) tolerates exactly this subtree. The native loader
# never opens it. It is admitted only as directories and single-link regular files, with the
# same ownership/ACL admission as every other member, and is never hashed or consumed.
HF_CACHE_ROOT = '.cache'
HF_CACHE = '.cache/huggingface'
REASON_CODE = re.compile(r'[A-Z][A-Z0-9_]{0,63}')


def reason_code(value):
    """The refusal code if ``value`` is exactly one (no paths, no free text), else None."""
    return value if isinstance(value, str) and REASON_CODE.fullmatch(value) else None


def in_hf_cache(rel):
    return rel in (HF_CACHE_ROOT, HF_CACHE) or rel.startswith(HF_CACHE + '/')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def absolute_path(value):
    require(isinstance(value, (str, os.PathLike)), 'NONCANONICAL_ABSOLUTE_PATH')
    value = os.fspath(value)
    require(isinstance(value, str) and value.startswith('/') and value != '/' and
            value == value.strip() and not any(ord(c) < 32 or ord(c) == 127 for c in value) and
            all(p not in ('', '.', '..') for p in value.split('/')[1:]), 'NONCANONICAL_ABSOLUTE_PATH')
    return value


def directory_identity(st):
    # Ignore ancestor mtime/size: unrelated siblings may be created during a scan.
    return (st.st_dev, st.st_ino, st.st_mode, st.st_uid, st.st_gid)


def identity(st):
    return (*directory_identity(st), st.st_nlink, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


# Linux filesystems with kernel POSIX DAC/ACL semantics. Do not infer absence of
# NFSv4/SMB/FUSE/overlay access policy from an absent POSIX ACL xattr.
POSIX_FILESYSTEMS = frozenset({0xEF53, 0x58465342, 0x9123683E, 0x01021994})
ACCESS_ACL = 'system.posix_acl_access'
UNDEFINED_ID = 0xffffffff


def filesystem_type(fd):
    require(sys.platform == 'linux', 'UNSUPPORTED_FILESYSTEM')
    # f_type is the first native long in Linux struct statfs. Reserve more than
    # the complete struct on both supported 64-bit ABIs; never pass a short struct.
    buffer = (ctypes.c_long * 32)()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.fstatfs(ctypes.c_int(fd), ctypes.byref(buffer)) != 0:
        raise ValueError('FILESYSTEM_VERIFICATION_FAILED')
    return buffer[0] & 0xffffffff


def read_access_acl(fd):
    try:
        names = os.listxattr(fd)
        # user.* is opaque data; the known Linux labels/integrity/capability
        # metadata below are not additional inode-write ACL mechanisms.
        known = {ACCESS_ACL, 'system.posix_acl_default', 'security.selinux',
                 'security.SMACK64', 'security.apparmor', 'security.ima',
                 'security.evm', 'security.capability'}
        require(all(n.startswith('user.') or n in known for n in names), 'UNSUPPORTED_ACCESS_MECHANISM')
        try:
            return os.getxattr(fd, ACCESS_ACL)
        except OSError as error:
            if error.errno == errno.ENODATA: return None
            raise
    except OSError as error:
        raise ValueError('ACCESS_ACL_VERIFICATION_FAILED') from error


def parse_acl(raw, mode):
    """Linux POSIX ACL xattr v2, including canonical order and mode projection."""
    require(isinstance(raw, bytes) and len(raw) >= 28 and (len(raw)-4) % 8 == 0 and
            struct.unpack_from('<I', raw)[0] == 2, 'INVALID_ACCESS_ACL')
    entries = tuple(struct.iter_unpack('<HHI', raw[4:]))
    require(all(tag in (1, 2, 4, 8, 16, 32) and perms <= 7 and
                ((ident != UNDEFINED_ID) if tag in (2, 8) else (ident == UNDEFINED_ID))
                for tag, perms, ident in entries), 'INVALID_ACCESS_ACL')
    require(len({(tag, ident) for tag, _, ident in entries}) == len(entries) and
            entries == tuple(sorted(entries, key=lambda e: (e[0], e[2]))), 'INVALID_ACCESS_ACL')
    base = {tag: perms for tag, perms, _ in entries if tag not in (2, 8)}
    require(set(base) in ({1, 4, 32}, {1, 4, 16, 32}) and
            (not any(tag in (2, 8) for tag, _, _ in entries) or 16 in base), 'INVALID_ACCESS_ACL')
    require((mode & 0o777) == (base[1] << 6 | base.get(16, base[4]) << 3 | base[32]),
            'ACCESS_ACL_MODE_MISMATCH')
    return entries, base.get(16, 7)


class AccessPolicy:
    def __init__(self, *, authority=False):
        self.uid = os.getuid()
        self.authority = authority
        self.trusted_uids = (0,) if authority else (0, self.uid)
        self.membership = None

    def private_group(self, owner, gid):
        if self.authority or owner != self.uid: return False
        if self.membership is None: self.membership = membership_snapshot()
        _, users, groups, memberships = self.membership
        owners = [u for u in users if u[1] == self.uid]
        if len(owners) != 1: return False
        name, uid, primary = owners[0]
        # systemd DynamicUser allocates from this reserved range. Never admit
        # one of those IDs as a durable private local group (even while idle).
        if 61184 <= gid <= 65519: return False
        if primary != gid or (name, gid) not in {(g[0], g[1]) for g in groups}: return False
        return {u for u, gids in memberships if gid in gids} == {uid}

    def recheck(self):
        require(os.getuid() == self.uid, 'MEMBERSHIP_CHANGED')
        if self.membership is not None:
            require(self.membership == membership_snapshot(), 'MEMBERSHIP_AUTHORITY_CHANGED')


def trusted_metadata(st, *, sticky=False, policy=None, acl=None):
    """Permission decision only; production callers use access_snapshot's FD binding."""
    own_policy = policy is None
    policy = policy or AccessPolicy()
    require(st.st_uid in policy.trusted_uids, 'UNSAFE_OWNER')
    trusted_temp = (sticky and not policy.authority and stat.S_ISDIR(st.st_mode) and
                    st.st_uid == 0 and bool(st.st_mode & stat.S_ISVTX))
    require(not st.st_mode & 0o002 or trusted_temp, 'UNSAFE_WRITE_PERMISSIONS')
    if acl is None:
        require(not st.st_mode & 0o020 or trusted_temp or policy.private_group(st.st_uid, st.st_gid),
                'UNSAFE_WRITE_PERMISSIONS')
    else:
        entries, mask = parse_acl(acl, st.st_mode)
        for tag, perms, ident in entries:
            if tag in (2, 4, 8): perms &= mask
            if not perms & 2 or tag in (1, 16): continue
            if tag == 2:
                safe = ident in policy.trusted_uids
            elif tag == 4:
                safe = trusted_temp or policy.private_group(st.st_uid, st.st_gid)
            elif tag == 8:
                safe = policy.private_group(policy.uid, ident)
            else:  # ACL_OTHER, never masked
                safe = trusted_temp
            require(safe, 'UNSAFE_WRITE_ACCESS_ACL')
    if own_policy: policy.recheck()


def access_snapshot(fd, policy, *, sticky=False):
    before = os.fstat(fd)
    fs = filesystem_type(fd)
    require(fs in POSIX_FILESYSTEMS, 'UNSUPPORTED_FILESYSTEM')
    acl = read_access_acl(fd)
    trusted_metadata(before, sticky=sticky, policy=policy, acl=acl)
    after = os.fstat(fd)
    # Directory ctime changes for unrelated sibling creations. Bind access
    # metadata, NOT full ancestor stat, and read the ACL back on the same inode.
    key = directory_identity if stat.S_ISDIR(before.st_mode) else identity
    require(key(before) == key(after) and acl == read_access_acl(fd), 'ACCESS_METADATA_CHANGED')
    require(key(before) == key(os.fstat(fd)), 'ACCESS_METADATA_CHANGED')
    return before, (directory_identity(before), fs, acl)


@contextmanager
def directory_fd(path, *, policy=None):
    path = absolute_path(path)
    policy = policy or AccessPolicy()
    fds = [os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)]
    chain = []
    try:
        _, root_access = access_snapshot(fds[0], policy)
        parts = path.split('/')[1:]
        for index, part in enumerate(parts):
            parent = fds[-1]
            fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            fds.append(fd)
            sticky = index < len(parts)-1
            _, access = access_snapshot(fd, policy, sticky=sticky)
            chain.append((parent, part, fd, access, sticky))
        yield fds[-1]
        require(root_access == access_snapshot(fds[0], policy)[1], 'ANCESTOR_CHANGED')
        for parent, part, fd, before, sticky in chain:
            held, access = access_snapshot(fd, policy, sticky=sticky)
            current = os.stat(part, dir_fd=parent, follow_symlinks=False)
            require(before == access and directory_identity(held) == directory_identity(current), 'ANCESTOR_CHANGED')
        policy.recheck()
    finally:
        for fd in reversed(fds): os.close(fd)


def _authority_file(name):
    # No NSS/group recursion at the trust anchor: root-only, no group writes.
    policy = AccessPolicy(authority=True)
    with directory_fd('/etc', policy=policy) as parent:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            before, access = access_snapshot(fd, policy)
            require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
                    0 < before.st_size <= 4 * 1024 * 1024, 'UNSAFE_MEMBERSHIP_AUTHORITY')
            data = bytearray()
            while len(data) <= before.st_size:
                block = os.read(fd, min(65536, before.st_size+1-len(data)))
                if not block: break
                data.extend(block)
            after, after_access = access_snapshot(fd, policy)
            require(len(data) == before.st_size and access == after_access and
                    identity(before) == identity(after) ==
                    identity(os.stat(name, dir_fd=parent, follow_symlinks=False)), 'MEMBERSHIP_AUTHORITY_CHANGED')
            return (identity(before), access), bytes(data)
        finally: os.close(fd)


def _systemd_sources():
    """Support files+systemd only with no supplemental userdb/provider state.

    NSS enumeration alone is NOT proof that a provider enumerates all principals.
    Refuse userdb drop-ins/unknown services and container mappings. The built-in
    DynamicUser socket alone is supported: NSS must enumerate no non-files users
    or groups, and the private GID must be outside systemd's dynamic range.
    """
    result = []
    for path in ('/etc/userdb', '/run/userdb', '/run/host/userdb', '/usr/lib/userdb',
                 '/run/systemd/userdb', '/run/systemd/dynamic-uid', '/run/systemd/machines'):
        policy = AccessPolicy(authority=True)
        with ExitStack() as stack:
            fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            stack.callback(os.close, fd)
            chain = [(fd, access_snapshot(fd, policy)[1])]
            missing = None
            for part in path.split('/')[1:]:
                try: child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                except FileNotFoundError:
                    missing = part; break
                stack.callback(os.close, child)
                fd = child
                chain.append((fd, access_snapshot(fd, policy)[1]))
            providers = ()
            if missing is None:
                names = sorted(os.listdir(fd))
                if path == '/run/systemd/userdb' and names == ['io.systemd.DynamicUser']:
                    info = os.stat(names[0], dir_fd=fd, follow_symlinks=False)
                    require(stat.S_ISSOCK(info.st_mode) and info.st_uid == 0, 'UNSAFE_MEMBERSHIP_AUTHORITY')
                    providers = ((names[0], identity(info)),)
                else:
                    require(not names, 'UNSUPPORTED_MEMBERSHIP_AUTHORITY')
                require(names == sorted(os.listdir(fd)), 'MEMBERSHIP_AUTHORITY_CHANGED')
            for held, before in chain:
                require(before == access_snapshot(held, policy)[1], 'MEMBERSHIP_AUTHORITY_CHANGED')
            result.append((path, tuple(access for _, access in chain), missing, providers))
    return tuple(result)


def membership_snapshot():
    """Frozen local authority plus independently cross-checked NSS memberships."""
    names = ('nsswitch.conf', 'passwd', 'group')
    sources = tuple(_authority_file(n) for n in names)
    nss, passwd, group = (value[1].decode('utf-8', errors='strict') for value in sources)
    databases = {}
    for line in nss.splitlines():
        line = line.split('#', 1)[0].strip()
        if not line: continue
        key, sep, rest = line.partition(':')
        if key.strip() in ('passwd', 'group', 'initgroups'):
            key = key.strip()
            require(sep and key not in databases, 'UNSUPPORTED_MEMBERSHIP_AUTHORITY')
            databases[key] = tuple(rest.split())
    supported = (('files',), ('files', 'systemd'))
    require(databases.get('passwd') in supported and databases.get('group') in supported and
            databases.get('initgroups', databases['group']) in supported, 'UNSUPPORTED_MEMBERSHIP_AUTHORITY')
    systemd = any('systemd' in v for v in databases.values())
    extra = _systemd_sources() if systemd else ()

    def rows(text, count):
        parsed = []
        for line in text.splitlines():
            if not line or line.startswith('#'): continue
            fields = line.split(':')
            require(len(fields) == count and fields[0] and fields[0][0] not in '+-' and
                    fields[0].isascii() and not any(c.isspace() for c in fields[0]), 'INVALID_MEMBERSHIP_AUTHORITY')
            parsed.append(fields)
        return parsed

    def number(value):
        require(value.isascii() and value.isdigit() and len(value) <= 10 and
                0 <= int(value) < UNDEFINED_ID, 'INVALID_MEMBERSHIP_AUTHORITY')
        return int(value)

    users = tuple(sorted((r[0], number(r[2]), number(r[3])) for r in rows(passwd, 7)))
    groups = tuple(sorted((r[0], number(r[2]), tuple(sorted(r[3].split(','))) if r[3] else ())
                          for r in rows(group, 4)))
    for records in (users, groups):
        require(len({r[0] for r in records}) == len(records) == len({r[1] for r in records}),
                'AMBIGUOUS_MEMBERSHIP_AUTHORITY')
    accounts = {u[0]: u[1] for u in users}
    require(all(len(set(g[2])) == len(g[2]) and all(n in accounts for n in g[2]) for g in groups),
            'UNKNOWN_GROUP_MEMBERSHIP')
    # Compare full enumerations AND initgroups, not just getgrgid().gr_mem.
    try:
        observed_users = tuple(sorted((p.pw_name, p.pw_uid, p.pw_gid) for p in pwd.getpwall()))
        observed_groups = tuple(sorted((g.gr_name, g.gr_gid, tuple(sorted(g.gr_mem))) for g in grp.getgrall()))
        require(users == observed_users and groups == observed_groups, 'NSS_MEMBERSHIP_AUTHORITY_MISMATCH')
        memberships = []
        for name, uid, primary in users:
            expected = tuple(sorted({primary, *(g[1] for g in groups if name in g[2])}))
            observed = tuple(sorted(set(os.getgrouplist(name, primary))))
            require(expected == observed, 'NSS_MEMBERSHIP_AUTHORITY_MISMATCH')
            memberships.append((uid, observed))
    except (KeyError, OSError) as error:
        raise ValueError('MEMBERSHIP_AUTHORITY_UNVERIFIABLE') from error
    require(sources == tuple(_authority_file(n) for n in names) and
            (not systemd or extra == _systemd_sources()), 'MEMBERSHIP_AUTHORITY_CHANGED')
    return (sources, extra), users, groups, tuple(memberships)


def hash_at(parent, name, expected=None, *, policy=None, allow_hardlinks=False):
    own_policy = policy is None
    policy = policy or AccessPolicy()
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    try:
        opened = os.fstat(fd)
        require(stat.S_ISREG(opened.st_mode) and
                (opened.st_nlink >= 1 if allow_hardlinks else opened.st_nlink == 1), 'NONREGULAR_OR_LINKED_FILE')
        before, access = access_snapshot(fd, policy)
        require(identity(opened) == identity(before), 'PAYLOAD_CHANGED')
        if expected is not None:
            require(before.st_size == expected['bytes'], 'PAYLOAD_SIZE_MISMATCH')
        digest = hashlib.sha256()
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block: break
            digest.update(block)
        after, after_access = access_snapshot(fd, policy)
        require(access == after_access and identity(before) == identity(after) ==
                identity(os.stat(name, dir_fd=parent, follow_symlinks=False)), 'PAYLOAD_CHANGED')
        if expected is not None:
            require(digest.hexdigest() == expected['sha256'], 'PAYLOAD_HASH_MISMATCH')
        if own_policy: policy.recheck()
        return identity(before), access
    finally:
        os.close(fd)


def verify_file(path, expected):
    """Executable/dependency verification always retains the single-link rule."""
    path = Path(absolute_path(path))
    policy = AccessPolicy()
    with directory_fd(path.parent, policy=policy) as fd:
        return hash_at(fd, path.name, expected, policy=policy)


def stat_at(parent, name, *, policy):
    """Link-count, type and access admission of a never-consumed file; no read, no hash."""
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    except OSError as error:  # a symlink (ELOOP) or an entry that vanished
        raise ValueError('NONREGULAR_OR_LINKED_FILE') from error
    try:
        opened = os.fstat(fd)
        require(stat.S_ISREG(opened.st_mode) and opened.st_nlink == 1, 'NONREGULAR_OR_LINKED_FILE')
        before, access = access_snapshot(fd, policy)
        require(identity(opened) == identity(before), 'PAYLOAD_CHANGED')
        return identity(before), access
    finally:
        os.close(fd)


def verify_payload(directory, pack):
    """Read-only model inventory/SHA admission; never used for write destinations."""
    expected = pack['files']
    require(all(isinstance(p, str) and not p.startswith('/') and
                all(x not in ('', '.', '..') for x in p.split('/')) for p in expected), 'INVALID_PAYLOAD_PATH')
    require(not any(p == HF_CACHE_ROOT or p.startswith(HF_CACHE_ROOT + '/') for p in expected), 'RESERVED_MODEL_PATH')
    policy = AccessPolicy()
    seen = set()
    directories, files = [], []
    with directory_fd(directory, policy=policy) as root, ExitStack() as stack:
        def walk(fd, prefix=''):
            info, access = access_snapshot(fd, policy)
            names = sorted(os.listdir(fd))
            directories.append((fd, identity(info), access, names))
            for name in names:
                rel = prefix + name
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    require(in_hf_cache(rel) or any(p.startswith(rel + '/') for p in expected), 'UNLISTED_MODEL_DIRECTORY')
                    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    stack.callback(os.close, child)
                    require(identity(info) == identity(os.fstat(child)), 'PAYLOAD_CHANGED')
                    walk(child, rel + '/')
                    files.append((fd, name, (identity(info), None)))
                elif rel.startswith(HF_CACHE + '/'):
                    files.append((fd, name, stat_at(fd, name, policy=policy)))
                else:
                    require(rel in expected or rel in MODEL_DOCUMENTS, 'UNLISTED_MODEL_PAYLOAD')
                    snapshot = hash_at(fd, name, expected.get(rel), policy=policy, allow_hardlinks=rel in expected)
                    files.append((fd, name, snapshot))
                    seen.add(rel)
        walk(root)
        # Recheck the whole tree, including earlier nested siblings after later
        # hashes. Reopen leaves no-follow, bind to the hashed inode and ACL bytes.
        for parent, name, (before, access) in files:
            require(before == identity(os.stat(name, dir_fd=parent, follow_symlinks=False)), 'PAYLOAD_CHANGED')
            if access is None: continue  # directory FD remains held below
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            try:
                info, current_access = access_snapshot(fd, policy)
                require(before == identity(info) and access == current_access, 'PAYLOAD_CHANGED')
            finally: os.close(fd)
        for fd, before, access, names in directories:
            info, current_access = access_snapshot(fd, policy)
            require(names == sorted(os.listdir(fd)) and before == identity(info) and access == current_access,
                    'PAYLOAD_CHANGED')
    require(set(expected) <= seen, 'MISSING_MODEL_PAYLOAD')


SSH_OPTIONS = ['-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
               '-o', 'ForwardAgent=no', '-o', 'ClearAllForwardings=yes', '-o', 'PermitLocalCommand=no',
               '-o', 'SendEnv=-*', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15',
               '-o', 'ServerAliveCountMax=2']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def accept_response(request, response):
    require(isinstance(response, dict) and set(response) == {'request_sha256', 'verified'} and
            response['verified'] is True and response['request_sha256'] == digest(request), 'RANK_IDENTITY_MISMATCH')


def bootstrap(helper, expected_hash, request):
    """Transmit the bundled helper, hash its exact decoded bytes BEFORE exec.

    No remote import path, user site, remote helper path, or credential forwarding.
    The SSH host/account and system Python are the trust anchors, not this hash.
    """
    source = ('import base64,hashlib; b=base64.b64decode(' + repr(base64.b64encode(helper).decode()) + '); '
              'assert hashlib.sha256(b).hexdigest()==' + repr(expected_hash) + '; '
              'exec(compile(b,"<verified-runtime-helper>","exec"),{"__name__":"__main__",'
              '"REQUEST":' + repr(request) + '})')
    return '/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 /usr/bin/python3 -I -S -B -c ' + shlex.quote(source)


def clean_env(explicit=None):
    env = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8',
           'HOME': pwd.getpwuid(os.getuid()).pw_dir}
    env.update(explicit or {})
    return env


def probe(argv, env=None):
    # Trusted bounded version/package probes, never model inference or a launcher.
    result = subprocess.run(argv, env=clean_env(env), stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    require(result.returncode == 0 and len(result.stdout) < 1024*1024, 'IDENTITY_PROBE_FAILED')
    return result.stdout


def verify_native(request):
    binary_dir = Path(absolute_path(request['binary_dir']))
    for name, item in request['binaries'].items():
        require(name in ('rexl3-serve', 'rexl3-cluster'), 'INVALID_BINARY_NAME')
        verify_file(binary_dir/name, item)
        with (binary_dir/name).open('rb') as f: header = f.read(20)
        require(header[:6] == b'\x7fELF\x02\x01' and header[18:20] == b'\xb7\0', 'NATIVE_ARCH_MISMATCH')
    version = {}
    for line in probe([str(binary_dir/'rexl3-serve'), '--version'], request['env']).splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and fields[0] not in version, 'INVALID_NATIVE_VERSION')
        version[fields[0]] = fields[1]
    require(version == request['native_version_fields'], 'NATIVE_VERSION_MISMATCH')
    # Match the immutable dependency lock's strictness; record/compare policy,
    # never silently pin a different version or install anything on a rank.
    for item in request['dependencies']['apt']:
        output = probe(['/usr/bin/dpkg-query', '-W', '-f=${Status}\t${Version}', item['package']]).strip().split('\t')
        require(len(output) == 2 and output[0] == 'install ok installed', 'DEPENDENCY_MISSING')
        if item['strict_version']: require(output[1] == item['version'], 'DEPENDENCY_VERSION_MISMATCH')
    return binary_dir/'rexl3-serve'


def state_directory(request, create=False):
    # Unique launch namespace; never share native pid files with a prior attempt.
    home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    path = home / request['state_relative'] / request['group']
    require(not Path(request['state_relative']).is_absolute() and
            all(p not in ('', '.', '..') for p in request['state_relative'].split('/')) and
            request['group'].isalnum(), 'INVALID_STATE_NAMESPACE')
    if create:
        # Descriptor-relative mkdir cannot follow an attacker planted symlink.
        with directory_fd(home) as homefd:
            fd = os.dup(homefd)
            try:
                for part in [*request['state_relative'].split('/'), request['group']]:
                    try: os.mkdir(part, 0o700, dir_fd=fd)
                    except FileExistsError: pass
                    nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    os.close(fd); fd = nxt
                    info = os.fstat(fd)
                    require(info.st_uid == os.getuid() and not info.st_mode & 0o022, 'UNSAFE_REMOTE_STATE')
            finally: os.close(fd)
    return path


@contextmanager
def saved_number(parent, name, error):
    """Native script.rs writes decimal /proc start ticks (not a boot UUID)."""
    f = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    try:
        policy = AccessPolicy()
        before, access = access_snapshot(f, policy)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 < before.st_size <= 32, error)
        raw = os.read(f, 33)
        # A single decimal field, with the optional newline emitted by stime/echo.
        value = raw.removesuffix(b'\n')
        require(value.isdigit() and len(raw) == before.st_size, error)
        yield value.decode('ascii')
        after, after_access = access_snapshot(f, policy)
        require(access == after_access and identity(before) == identity(after) ==
                identity(os.stat(name, dir_fd=parent, follow_symlinks=False)), 'RANK_STATE_CHANGED')
        policy.recheck()
    finally:
        os.close(f)


def verify_listener(request):
    directory = state_directory(request)
    rank = request['rank']
    with directory_fd(directory) as fd, \
         saved_number(fd, 'rank'+str(rank)+'.pid', 'INVALID_RANK_PID') as raw, \
         saved_number(fd, 'rank'+str(rank)+'.start', 'INVALID_RANK_START') as saved:
        pid = int(raw)
        require(pid > 1, 'INVALID_RANK_PID')
        proc = Path('/proc')/str(pid)
        before = (proc/'stat').read_text().rsplit(')',1)[1].split()[19]
        require(saved == before, 'RANK_SAVED_LIFETIME_MISMATCH')
        binary = Path(request['binary_dir'])/'rexl3-serve'
        require(os.readlink(proc/'exe') == str(binary) and proc.stat().st_uid == os.getuid(), 'WRONG_RANK_PROCESS')
        args = (proc/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        require(args == [str(binary), *request['argv']], 'WRONG_RANK_ARGV')
        environment = dict(item.split('=', 1) for item in (proc/'environ').read_bytes().decode().split('\0') if '=' in item)
        base_keys = {'PATH', 'HOME', 'LANG', 'LC_ALL', 'SHLVL', 'PWD', '_'}
        require(set(environment) <= base_keys | set(request['env']) and
                all(environment.get(k) == v for k, v in request['env'].items()), 'RANK_ENVIRONMENT_MISMATCH')
        if rank == 0:
            inodes = set()
            for kind in ('tcp', 'tcp6'):
                for line in (proc/'net'/kind).read_text().splitlines()[1:]:
                    fields = line.split()
                    if fields[3] == '0A' and int(fields[1].rsplit(':',1)[1], 16) == request['http_port']: inodes.add(fields[9])
            owned = set()
            for f in (proc/'fd').iterdir():
                try: target = os.readlink(f)
                except OSError: continue
                if target.startswith('socket:['): owned.add(target[8:-1])
            require(bool(inodes) and inodes <= owned, 'WRONG_LEADER_LISTENER')
        require(saved == before == (proc/'stat').read_text().rsplit(')',1)[1].split()[19], 'RANK_PROCESS_CHANGED')


def attest(request):
    if request['operation'] == 'ownership':
        verify_listener(request)
    else:
        binary = verify_native(request)
        verify_payload(request['model_dir'], request['pack'])
        index_name = 'model.safetensors.index.json'
        if index_name in request['pack']['files']:
            index = json.loads((Path(request['model_dir'])/index_name).read_text())
            shards = set(index['weight_map'].values())
            require(shards == {p for p in request['pack']['files'] if p.endswith('.safetensors')}, 'MODEL_INDEX_MISMATCH')
        lines = probe([str(binary), '--print-fingerprints', '--model', request['model_dir']], request['env']).splitlines()
        require([line.split() for line in lines if line.split() and line.split()[0] == 'target'] ==
                [['target', request['pack']['fingerprint']]], 'MODEL_FINGERPRINT_MISMATCH')
        if request['operation'] == 'prepare': state_directory(request, create=True)
    return {'request_sha256': digest(request), 'verified': True}


if __name__ == '__main__':
    try: print(json.dumps(attest(globals()['REQUEST']), sort_keys=True))
    except Exception as error:
        # Only a bare refusal CODE crosses the SSH boundary: never a path, host or free text.
        code = reason_code(str(error)) if isinstance(error, ValueError) else None
        print('REMOTE_IDENTITY_REFUSED' + (': ' + code if code else ''), file=sys.stderr)
        raise SystemExit(23)
