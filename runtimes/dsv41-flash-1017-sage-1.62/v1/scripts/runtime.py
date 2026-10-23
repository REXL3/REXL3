#!/usr/bin/env python3
"""Shared runtime lifecycle; start is FOREGROUND, not a daemon.

Retains MiniCPM with_server.py's owned Popen/finally cleanup design. A lifetime
advisory lock and a process-start identity protect separate status/stop calls.
Ctrl-C/SIGTERM drains only this launch; stop requests the verified foreground
owner to drain, never signals a PID file's server PID. Linux uses pidfds when
available to pin/recheck owner identity before writing a run-specific request.

Cluster operations delegate to shipped rexl3-cluster (which owns node GPU locks).
V1 accepts immutable, already site-resolved local-node JSON configurations. Remote
ranks are refused until the launcher attests their locked payload/sidecars and
scrubs remote credentials: today's check only compares cross-rank metadata. Site
rewrites beyond single-node HTTP placeholders are refused, never silently ignored.
Logs are private, no-follow by default; diagnostics never includes raw logs,
environment, requests/responses, hostnames or filesystem paths.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack
import http.client
import ipaddress
import json
import os
from pathlib import Path
import queue
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import ContractError, canonical_digest, load_json, render_argv, safe_file, verify_release
from fetch_models import (RuntimeFailure, advisory_lock, atomic_json, clean_environment, demand, lock_held,
                          model_paths, prepare_models, private_open, run_owned, secure_directory, state_root)

ROOT = Path(__file__).resolve().parents[1]
STOP_GRACE_SECONDS = 5
STOP_WAIT_SECONDS = 35
ACTIVE_PHASES = {'preparing', 'starting', 'ready', 'stopping'}
PHASES = ACTIVE_PHASES | {'stopped', 'failed', 'degraded', 'unverified'}


class StopToken(threading.Event):
    """Durable run-specific request: duplicate stop calls send no TERM signals.

    Restoring a supervisor's signal handlers cannot race a second TERM into
    killing it halfway through its terminal state write.
    """
    def __init__(self, path, run_id):
        super().__init__()
        self.path, self.run_id = path, run_id

    def is_set(self):
        if not super().is_set() and (self.path.exists() or self.path.is_symlink()):
            demand(not self.path.is_symlink(), 22, 'UNSAFE_STOP_REQUEST')
            request = load_json(self.path)
            if isinstance(request, dict) and request.get('run_id') == self.run_id:
                self.set()
        return super().is_set()

    def wait(self, timeout=None):
        if self.is_set():
            return True
        super().wait(timeout)
        return self.is_set()


def process_identity(pid):
    """Return a boot/start identity, not a command-line substring or kill(pid, 0)."""
    if type(pid) is not int or pid <= 1:
        return None
    try:
        if sys.platform.startswith('linux'):
            directory = Path('/proc') / str(pid)
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                return None
            return {'pid': pid, 'start': Path('/proc/sys/kernel/random/boot_id').read_text().strip() + ':' + fields[19],
                    'group': int(fields[2]), 'uid': directory.stat().st_uid}
        # CPU fixture portability. The release's Linux path above has boot + ticks
        # and pidfd protection; non-Linux does not advertise a GPU release.
        result = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'uid=', '-o', 'pgid=', '-o', 'stat=', '-o', 'lstart='],
                                capture_output=True, text=True, timeout=2, env=clean_environment())
        fields = result.stdout.split()
        if result.returncode or len(fields) < 8 or fields[2].startswith('Z'):
            return None
        return {'pid': pid, 'uid': int(fields[0]), 'group': int(fields[1]), 'start': ' '.join(fields[3:])}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def same_process(identity):
    return isinstance(identity, dict) and identity.get('uid') == os.getuid() and process_identity(identity.get('pid')) == identity


def signal_identity(identity, sig):
    """Acquire pidfd BEFORE revalidating: PID reuse cannot redirect the signal."""
    demand(isinstance(identity, dict), 22, 'STALE_OR_REUSED_PID_REFUSED')
    pid = identity.get('pid')
    demand(type(pid) is int and pid > 1, 22, 'STALE_OR_REUSED_PID_REFUSED')
    fd = None
    try:
        if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
            try:
                fd = os.pidfd_open(pid, 0)
            except ProcessLookupError:
                return False
            except OSError as error:
                # Old kernels may lack pidfd despite a new Python. Permissions or
                # resource exhaustion are not grounds for a less safe fallback.
                import errno
                if error.errno not in (errno.ENOSYS, errno.EINVAL):
                    raise RuntimeFailure(22, 'PROCESS_SIGNAL_REFUSED') from error
        demand(same_process(identity), 22, 'STALE_OR_REUSED_PID_REFUSED')
        if fd is not None:
            signal.pidfd_send_signal(fd, sig, None, 0)
        else:
            os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return False
    finally:
        if fd is not None:
            os.close(fd)


def listener_owned(pid, port):
    """Require the listening socket to belong to the launched server, not HTTP 200."""
    try:
        if sys.platform.startswith('linux'):
            inodes = set()
            for name in ('tcp', 'tcp6'):
                source = Path('/proc') / str(pid) / 'net' / name
                if not source.exists():
                    continue
                for line in source.read_text().splitlines()[1:]:
                    fields = line.split()
                    if len(fields) > 9 and fields[3] == '0A' and int(fields[1].rsplit(':', 1)[1], 16) == port:
                        inodes.add(fields[9])
            owned = set()
            for fd in (Path('/proc') / str(pid) / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                    if target.startswith('socket:['):
                        owned.add(target[8:-1])
                except OSError:
                    continue
            return bool(inodes) and inodes <= owned
        result = subprocess.run(['/usr/sbin/lsof', '-nP', '-iTCP:' + str(port), '-sTCP:LISTEN', '-Fp'],
                                capture_output=True, text=True, timeout=2, env=clean_environment())
        owners = {int(line[1:]) for line in result.stdout.splitlines() if line.startswith('p')}
        return result.returncode == 0 and owners == {pid}
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def check_free_port(host, port):
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        for family, kind, protocol, _, address in addresses:
            with socket.socket(family, kind, protocol) as sock:
                # REUSEADDR permits a legitimate restart through TIME_WAIT, not a
                # competing listening socket. Never set SO_REUSEPORT.
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(address)
    except OSError as error:
        raise RuntimeFailure(22, 'PORT_OCCUPIED_OR_UNAVAILABLE') from error


def flag_value(words, flag):
    values = []
    for index, word in enumerate(words):
        if word == flag:
            demand(index + 1 < len(words), 2, 'MISSING_PROFILE_FLAG_VALUE')
            values.append(words[index + 1])
        elif word.startswith(flag + '='):
            values.append(word.split('=', 1)[1])
    demand(len(values) <= 1, 2, 'DUPLICATE_PROFILE_FLAG')
    return values[0] if values else None


class Configuration:
    def __init__(self, root, name=None, site=None):
        self.root = Path(root)
        self.profile, self.models, self.release = verify_release(self.root)
        self.name = name or self.profile['default_profile']
        demand(self.name in self.profile['profiles'], 2, 'UNKNOWN_PROFILE')
        self.choice = self.profile['profiles'][self.name]
        self.state = state_root(self.profile)
        self.kind = self.choice['kind']
        self.cancel = None
        self.site = {}
        if site:
            self.site = load_json(site)
            demand(isinstance(self.site, dict), 2, 'INVALID_SITE_OVERLAY')
            demand(set(self.site) <= set(self.choice['site_fields']), 2, 'UNAPPROVED_SITE_FIELD')
            supported = {'http_host', 'http_port'} if self.kind == 'single' else set()
            demand(set(self.site) <= supported, 2, 'UNSUPPORTED_SITE_OVERRIDE: use a reviewed, fully resolved cluster config; external model directories are not supported')
        self.host = self.site.get('http_host', self.choice.get('http_host', '127.0.0.1'))
        self.port = self.site.get('http_port', self.choice['http_port'])
        demand(type(self.port) is int and 1 <= self.port <= 65535, 2, 'INVALID_HTTP_PORT')
        # No DNS rebinding, URL credentials, redirects or inherited proxy settings.
        try:
            address = ipaddress.ip_address(self.host)
        except ValueError as error:
            raise RuntimeFailure(2, 'HTTP_HOST_MUST_BE_LITERAL_ADDRESS') from error
        demand(not address.is_unspecified and not address.is_multicast, 2, 'UNSAFE_HTTP_BINDING')
        if self.kind == 'single':
            demand(address.is_loopback, 2, 'UNSUPPORTED_NONLOOPBACK_SINGLE_BINDING')
        paths = model_paths(self.state, self.models)
        substitutions = {'@MODEL_DIR@': str(paths['target']), '@BUNDLE_DIR@': str(self.root), '@STATE_DIR@': str(self.state)}
        if 'draft' in paths:
            substitutions['@DRAFT_DIR@'] = str(paths['draft'])
        for key, value in (('http_host', self.host), ('http_port', self.port)):
            if key in self.choice['site_fields']:
                substitutions['@' + key.upper() + '@'] = str(value)
        words = render_argv(self.profile, self.name, substitutions)
        self.args = words
        self.env = {}
        for key, value in self.choice['env'].items():
            for token, replacement in substitutions.items():
                value = value.replace(token, replacement)
            demand(not re.search(r'@[A-Z][A-Z0-9_]*@', value), 2, 'UNRESOLVED_ENV_TOKEN')
            self.env[key] = value
        self.child_env = clean_environment(self.env)
        self.argv = [str(safe_file(self.root, 'bin/rexl3-serve')), *words]
        if self.kind == 'single':
            demand(flag_value(words, '--model') == str(paths['target']), 2, 'MODEL_PATH_MUST_BE_MANAGED')
            draft = flag_value(words, '--draft')
            demand(draft == (str(paths['draft']) if 'draft' in paths else None), 2, 'DRAFT_PATH_MISMATCH')
            demand(flag_value(words, '--port') == str(self.port), 2, 'HTTP_PORT_ARGV_MISMATCH')
            bound_host = flag_value(words, '--host')
            demand(bound_host == self.host, 2, 'HTTP_HOST_ARGV_MISMATCH: bind explicitly')
            for key in self.site:
                token = '@' + key.upper() + '@'
                demand(any(token in word for word in self.choice['argv']), 2, 'UNUSED_SITE_OVERRIDE')
        else:
            self.cluster_file = safe_file(self.root, self.choice['cluster_file'])
            self.group = self.choice['group']
            cluster = load_json(self.cluster_file)
            groups = [g for g in cluster.get('groups', []) if g.get('name') == self.group]
            demand(len(groups) == 1, 2, 'CLUSTER_GROUP_MISSING_OR_AMBIGUOUS')
            group = groups[0]
            demand(group.get('http_port') == self.port and group.get('http_host', '127.0.0.1') == self.host,
                   2, 'CLUSTER_HTTP_BINDING_MISMATCH')
            # No fallback to a historical example or an unpinned external env file.
            self.check_cluster_data(cluster, group, paths)
            self.argv = self.cluster_argv('up')
        self.identity = {'profile': self.name, 'profile_digest': canonical_digest(self.profile),
                         'model_lock_digest': canonical_digest(self.models), 'release_digest': canonical_digest(self.release),
                         'site_digest': canonical_digest(self.site)}

    def check_cluster_data(self, cluster, group, paths):
        demand(cluster.get('binary') == str(self.root / 'bin/rexl3-serve') and cluster.get('pack') == str(paths['target']),
               2, 'CLUSTER_MUST_USE_PINNED_LOCAL_BINARY_AND_MANAGED_PACK')
        ranks = group.get('ranks')
        demand(isinstance(ranks, list) and ranks and all(isinstance(name, str) for name in ranks),
               2, 'UNSUPPORTED_CLUSTER_RANK_MAP: use ordered node names')
        nodes = {node['name']: node for node in cluster.get('nodes', [])}
        demand(all(name in nodes for name in ranks), 2, 'CLUSTER_NODE_MISSING')
        demand(all(nodes[name].get('local') is True for name in ranks), 2,
               'CLUSTER_REMOTE_IDENTITY_UNSUPPORTED: remote payload attestation and credential scrubbing require a qualified launcher adapter')
        serve = cluster.get('serve', {})
        tp = cluster.get('tp', {})
        leader = nodes[ranks[0]]
        # Compare, do not choose serving values or infer the launcher's defaults.
        # This is schema.rs::rank_args(rank=0), not a second launch implementation.
        args = ['--model', cluster['pack'], *serve.get('flags', []), *leader.get('flags', []),
                *group.get('leader_flags', []), '--host', self.host, '--port', str(self.port)]
        env = {}
        if len(ranks) > 1:
            for key in ('connect_timeout_s', 'ready_timeout_s', 'peer_timeout_s'):
                demand(type(tp.get(key)) is int, 2, 'CLUSTER_TIMEOUT_MUST_BE_EXPLICIT')
            args += ['--tp-ranks', str(len(ranks)), '--tp-rank', '0', '--tp-peer',
                     leader.get('fabric_address', leader['address']) + ':' + str(group['tp_port']),
                     '--tp-connect-timeout', str(tp['connect_timeout_s']), '--tp-ready-timeout', str(tp['ready_timeout_s']),
                     '--tp-peer-timeout', str(tp['peer_timeout_s'])]
            nccl = cluster.get('nccl', {})
            demand(type(nccl.get('max_nchannels')) is int, 2, 'CLUSTER_NCCL_MUST_BE_EXPLICIT')
            env['NCCL_MAX_NCHANNELS'] = str(nccl['max_nchannels'])
            if nccl.get('socket_ifname'):
                env['NCCL_SOCKET_IFNAME'] = nccl['socket_ifname']
            if nccl.get('hca'):
                env['NCCL_IB_HCA'] = '=' + ','.join(nccl['hca'])
            env.update(nccl.get('env', {}))
            if nccl.get('lib'):
                env['REXL3_NCCL_LIB'] = nccl['lib']
        demand(type(tp.get('shutdown_grace_s')) is int, 2, 'CLUSTER_TIMEOUT_MUST_BE_EXPLICIT')
        args += ['--shutdown-grace', str(tp['shutdown_grace_s'])]
        env.update(serve.get('env', {})); env.update(leader.get('env', {}))
        demand(args == self.args and env == self.env, 2, 'CLUSTER_PROFILE_ARGV_OR_ENV_MISMATCH')
        def inspect(value):
            if isinstance(value, list):
                for item in value:
                    inspect(item)
            elif isinstance(value, dict):
                for key, item in value.items():
                    if key in ('env_file',) and item:
                        raise RuntimeFailure(2, 'UNSUPPORTED_CLUSTER_ENV_FILE: capture explicit environment instead')
                    if key in ('env', 'extra'):
                        demand(isinstance(item, dict), 2, 'INVALID_CLUSTER_ENV')
                        for name in item:
                            demand(not re.search(r'TOKEN|SECRET|PASSWORD|PRIVATE_KEY|API_KEY|CREDENTIAL|AUTHORIZATION', name)
                                   and name not in ('EXTRA_ARGS', 'LD_PRELOAD', 'PYTHONPATH', 'PYTHONHOME', 'REXL3_NO_CUDA', 'GB_NO_CUDA'),
                                   2, 'UNSAFE_CLUSTER_ENV')
                            if name.endswith('_KNOBS_FILE'):
                                demand(item[name].startswith(str(self.root) + '/') and
                                       Path(item[name]).relative_to(self.root).as_posix() in self.choice['files'],
                                       2, 'CLUSTER_KNOBS_NOT_PINNED')
                    inspect(item)
            elif isinstance(value, str):
                demand('--allow-unvalidated-pack' not in value and '--skip-check' not in value
                       and not re.search(r'@[A-Z][A-Z0-9_]*@', value), 2, 'UNSAFE_OR_UNRESOLVED_CLUSTER_CONFIG')
        inspect(cluster)

    def cluster_argv(self, command, extra=()):
        return [str(safe_file(self.root, 'bin/rexl3-cluster')), command, '--cluster', str(self.cluster_file), '--group', self.group, *extra]

    def cluster_call(self, command, extra=(), timeout=30, log=None, cancel=None):
        try:
            result = run_owned(self.cluster_argv(command, extra), env=self.child_env,
                               stdout=log if log is not None else subprocess.PIPE, timeout=timeout, cancel=cancel)
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeFailure(24, 'CLUSTER_OPERATION_FAILED: ' + command) from error
        return result

    def read_state(self):
        path = self.state / 'instance.json'
        if not path.exists() and not path.is_symlink():
            return None
        demand(not path.is_symlink(), 22, 'UNSAFE_INSTANCE_STATE')
        record = load_json(path)
        demand(isinstance(record, dict) and record.get('schema') == 'rexl3.instance.v1'
               and record.get('phase') in PHASES, 22, 'INVALID_INSTANCE_STATE')
        return record

    def matching_state(self, record):
        demand(all(record.get(k) == v for k, v in self.identity.items()), 22, 'INSTANCE_PROFILE_OR_RELEASE_MISMATCH')

    def output(self):
        return {'argv': self.argv, 'env': self.env, 'profile': self.name, 'kind': self.kind,
                'http_host': self.host, 'http_port': self.port, 'foreground': True, **self.identity}


def request(config, path, deadline, payload=None):
    """A wall-clock bound, including a peer that trickles headers forever."""
    remaining = deadline - time.monotonic()
    demand(remaining > 0, 24, 'READINESS_TIMEOUT')
    result = queue.Queue(maxsize=1)

    def worker():
        connection = http.client.HTTPConnection(config.host, config.port, timeout=min(remaining, 2))
        try:
            body = None if payload is None else json.dumps(payload).encode()
            connection.request('GET' if body is None else 'POST', path, body=body,
                               headers={'Content-Type': 'application/json', 'Connection': 'close'})
            response = connection.getresponse()
            data = bytearray()
            while True:
                chunk = response.read1(16384)
                if not chunk:
                    break
                data.extend(chunk)
                demand(len(data) <= 1024 * 1024 and time.monotonic() < deadline, 24, 'INVALID_OR_UNBOUNDED_RESPONSE')
            demand(response.status == 200, 24, 'API_HTTP_FAILURE')
            result.put((True, bytes(data)))
        except Exception:
            result.put((False, None))
        finally:
            connection.close()

    threading.Thread(target=worker, daemon=True).start()
    while True:
        demand(config.cancel is None or not config.cancel.is_set(), 24, 'OPERATION_INTERRUPTED')
        remaining = deadline - time.monotonic()
        demand(remaining > 0, 24, 'READINESS_TIMEOUT')
        try:
            ok, body = result.get(timeout=min(.2, remaining))
            break
        except queue.Empty:
            continue
    demand(ok, 24, 'API_REQUEST_FAILED')
    return body


def api_identity(config, deadline):
    request(config, '/health/ready', deadline)
    try:
        data = json.loads(request(config, '/v1/models', deadline))
        entries = data['data']
        demand(isinstance(entries, list) and len(entries) == 1, 24, 'MODEL_IDENTITY_MISMATCH')
        model = entries[0]
        demand(model['id'] == config.choice['smoke_request']['model'], 24, 'MODEL_IDENTITY_MISMATCH')
        validated = model.get('pack_validated')
        demand(isinstance(validated, str) and validated and not validated.lower().startswith(('no', 'unknown', 'unvalidated'))
               and model.get('pack_warning') is None, 24, 'MODEL_NOT_VALIDATED')
        configured = model.get('configured_max_model_len')
        effective = model.get('max_model_len')
        demand(type(configured) is int and type(effective) is int and 0 < effective <= configured, 24, 'MODEL_LIMIT_INVALID')
        expected = flag_value(config.args, '--max-model-len')
        if expected is not None:
            demand(configured == int(expected), 24, 'MODEL_LIMIT_MISMATCH')
    except (KeyError, TypeError, ValueError, UnicodeError) as error:
        raise RuntimeFailure(24, 'MODEL_IDENTITY_RESPONSE_INVALID') from error


def validate_usage(usage):
    demand(isinstance(usage, dict), 24, 'SMOKE_USAGE_MISSING')
    fields = [usage.get(key) for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')]
    demand(all(type(n) is int and n > 0 for n in fields) and fields[0] + fields[1] == fields[2], 24, 'SMOKE_USAGE_INVALID')


def smoke(config, deadline):
    payload = config.choice['smoke_request']
    try:
        response = json.loads(request(config, '/v1/chat/completions', deadline, payload))
        demand(response.get('object') == 'chat.completion' and response.get('model') == payload['model'], 24, 'SMOKE_IDENTITY_INVALID')
        choices = response['choices']
        demand(isinstance(choices, list) and len(choices) == 1, 24, 'SMOKE_CHOICES_INVALID')
        choice = choices[0]
        demand(choice.get('finish_reason') == 'stop' and choice['message'].get('role') == 'assistant', 24, 'SMOKE_NOT_COMPLETE')
        content = choice['message']['content']
        demand(isinstance(content, str) and bool(content.strip()), 24, 'SMOKE_EMPTY_RESPONSE')
        if 'smoke_expect' in config.choice:
            demand(content.strip() == config.choice['smoke_expect'], 24, 'SMOKE_CONTENT_MISMATCH')
        validate_usage(response.get('usage'))
        if config.choice['capabilities']['sse']:
            streamed = {**payload, 'stream': True, 'stream_options': {'include_usage': True}}
            raw = request(config, '/v1/chat/completions', deadline, streamed).decode('utf-8')
            done = finished = False
            text = ''
            usage = None
            for frame in raw.replace('\r\n', '\n').split('\n\n'):
                rows = [line[5:].strip() for line in frame.splitlines() if line.startswith('data:')]
                if not rows:
                    continue
                demand(not done, 24, 'SMOKE_SSE_AFTER_DONE')
                value = '\n'.join(rows)
                if value == '[DONE]':
                    done = True
                    continue
                event = json.loads(value)
                demand(event.get('model') == payload['model'] and event.get('object') == 'chat.completion.chunk', 24, 'SMOKE_SSE_IDENTITY_INVALID')
                if event.get('usage') is not None:
                    usage = event['usage']
                for item in event['choices']:
                    demand(item.get('index') == 0, 24, 'SMOKE_SSE_CHOICE_INVALID')
                    text += item.get('delta', {}).get('content') or ''
                    if item.get('finish_reason') is not None:
                        demand(item['finish_reason'] == 'stop', 24, 'SMOKE_SSE_NOT_COMPLETE')
                        finished = True
            demand(done and finished and bool(text.strip()), 24, 'SMOKE_SSE_INCOMPLETE')
            validate_usage(usage)
            if 'smoke_expect' in config.choice:
                demand(text.strip() == config.choice['smoke_expect'], 24, 'SMOKE_CONTENT_MISMATCH')
    except (KeyError, TypeError, ValueError, UnicodeError) as error:
        raise RuntimeFailure(24, 'SMOKE_RESPONSE_INVALID') from error


def ensure_no_live_record(record):
    if record:
        for key in ('owner', 'child'):
            identity = record.get(key)
            if isinstance(identity, dict):
                # Even a reused PID is refused, never signalled or automatically erased.
                demand(process_identity(identity.get('pid')) is None, 22, 'STALE_OR_REUSED_PID_REFUSED')
        demand(not record.get('cluster_active'), 22, 'ORPHANED_CLUSTER_REFUSED: inspect with the recorded cluster launcher')


def terminate_owned(proc):
    if proc is None:
        return
    if proc.poll() is None:
        # This unreaped Popen child owns its session/group; no PID-file lookup.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=STOP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)


def start(config, accept_download=False):
    import doctor
    secure_directory(config.state)
    with ExitStack() as stack:
        lifetime = stack.enter_context(advisory_lock(config.state / 'instance.lock'))
        ensure_no_live_record(config.read_state())
        if config.kind == 'single':
            # Same conventional node lock as the existing MiniCPM/cluster tooling.
            gpu_lock = stack.enter_context(advisory_lock(doctor.GPU_LOCK_PATH, 'GPU_LOCK_BUSY', node_lock=True))
        else:
            gpu_lock = None  # rexl3-cluster owns ordered node locks; never double-acquire.
        check_free_port(config.host, config.port) if config.kind == 'single' else None
        run_id = uuid.uuid4().hex
        stopped = StopToken(config.state / 'stop-request.json', run_id)
        config.cancel = stopped
        previous = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, lambda *_: stopped.set())
        proc = None
        cluster_attempted = False
        record = {'schema': 'rexl3.instance.v1', **config.identity, 'run_id': run_id, 'phase': 'preparing',
                  'kind': config.kind, 'owner': process_identity(os.getpid()), 'child': None, 'cluster_active': False,
                  'server_exit_code': None, 'smoke_passed': False}
        demand(record['owner'] is not None, 22, 'PROCESS_IDENTITY_UNAVAILABLE')
        atomic_json(config.state / 'instance.json', record)
        log = None
        rc = 24
        try:
            import doctor
            # Static doctor is profile-independent; this adapter checks the selected
            # site's binding, argv, ownership and payload separately.
            doctor_rc = doctor.main([], root=config.root)
            demand(doctor_rc == 0, doctor_rc or 20, 'PREFLIGHT_FAILED')
            if config.kind == 'single':
                demand(doctor.gpu_idle(), 22, 'GPU_BUSY_OR_IDLE_PROBE_FAILED')
            prepare_models(config.root, accept_download=accept_download, cancel=stopped)
            if stopped.is_set():
                rc = 0
                return rc
            secure_directory(config.state / 'logs')
            log = stack.enter_context(private_open(config.state / 'logs' / (run_id + '.log'), 'x'))
            record['phase'] = 'starting'
            atomic_json(config.state / 'instance.json', record)
            deadline = time.monotonic() + config.choice['readiness_timeout_s']
            if config.kind == 'single':
                # Downloads may be long: recheck non-cooperating compute jobs.
                demand(doctor.gpu_idle(), 22, 'GPU_BUSY_OR_IDLE_PROBE_FAILED')
                check_free_port(config.host, config.port)
                # Inherit both locks: a hard-killed wrapper cannot allow another model
                # while its server is still alive. Normal children close them on exit.
                proc = subprocess.Popen(config.argv, env=config.child_env, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                        pass_fds=(lifetime.fileno(), gpu_lock.fileno()))
                record['child'] = process_identity(proc.pid)
                demand(record['child'] is not None, 24, 'CHILD_EXITED_BEFORE_READY')
            else:
                for command in ('validate', 'check'):
                    result = config.cluster_call(command, timeout=max(.1, deadline-time.monotonic()), log=log, cancel=stopped)
                    demand(result.returncode == 0, 24, 'CLUSTER_' + command.upper() + '_FAILED')
                cluster_attempted = True
                record['cluster_active'] = True
                atomic_json(config.state / 'instance.json', record)
                result = config.cluster_call('up', timeout=max(.1, deadline-time.monotonic()), log=log, cancel=stopped)
                demand(result.returncode == 0, 24, 'CLUSTER_UP_FAILED')
            atomic_json(config.state / 'instance.json', record)
            while not stopped.is_set():
                demand(time.monotonic() < deadline, 24, 'READINESS_TIMEOUT')
                if proc is not None:
                    demand(proc.poll() is None and same_process(record['child']), 24, 'CHILD_EXITED_BEFORE_READY')
                    if not listener_owned(proc.pid, config.port):
                        stopped.wait(.05)
                        continue
                else:
                    demand(config.cluster_call('status', timeout=max(.1, deadline-time.monotonic()), log=log, cancel=stopped).returncode == 0,
                           24, 'CLUSTER_STATUS_FAILED')
                try:
                    api_identity(config, min(deadline, time.monotonic()+1))
                    break
                except RuntimeFailure:
                    stopped.wait(.05)
            if stopped.is_set():
                rc = 0
                return rc
            smoke(config, deadline)
            if proc is not None:
                demand(proc.poll() is None and same_process(record['child']) and listener_owned(proc.pid, config.port),
                       24, 'OWNERSHIP_LOST_BEFORE_READY')
            record.update(phase='ready', smoke_passed=True)
            atomic_json(config.state / 'instance.json', record)
            host = '[' + config.host + ']' if ':' in config.host else config.host
            print('READY: http://' + host + ':' + str(config.port) + ' (foreground; Ctrl-C or launch.sh stop to stop)', flush=True)
            while not stopped.wait(.2):
                if proc is not None:
                    if proc.poll() is not None:
                        record['server_exit_code'] = proc.returncode
                        demand(proc.returncode == 0, 24, 'SERVER_EXITED')
                        break
                else:
                    demand(config.cluster_call('status', timeout=10, log=log, cancel=stopped).returncode == 0, 24, 'CLUSTER_DEGRADED')
                    stopped.wait(.8)
            rc = 0
            return rc
        except RuntimeFailure:
            if stopped.is_set():
                rc = 0
                return rc
            raise
        finally:
            # A full disk or failed state write must NEVER bypass child cleanup.
            clean = False
            try:
                record['phase'] = 'stopping'
                atomic_json(config.state / 'instance.json', record)
            finally:
                try:
                    terminate_owned(proc)
                    if proc is not None:
                        record['server_exit_code'] = proc.returncode
                    if cluster_attempted:
                        down = config.cluster_call('down', timeout=STOP_WAIT_SECONDS - 5, log=log)
                        demand(down.returncode == 0, 24, 'CLUSTER_CLEANUP_FAILED')
                    clean = True
                finally:
                    for sig, handler in previous.items():
                        signal.signal(sig, handler)
                    record['phase'] = 'stopped' if rc == 0 and clean else 'failed'
                    if clean:
                        record.update(owner=None, child=None, cluster_active=False)
                    atomic_json(config.state / 'instance.json', record)


def status(config):
    record = config.read_state()
    if not record:
        print(json.dumps({'phase': 'stopped', **config.identity})); return 24
    config.matching_state(record)
    phase = record['phase']
    if phase in ACTIVE_PHASES:
        demand(lock_held(config.state / 'instance.lock') and same_process(record.get('owner')), 22, 'STALE_OR_REUSED_PID_REFUSED')
        if phase == 'ready':
            if config.kind == 'single':
                demand(same_process(record.get('child')) and listener_owned(record['child']['pid'], config.port), 22, 'SERVER_OWNERSHIP_MISMATCH')
            else:
                demand(config.cluster_call('status').returncode == 0, 24, 'CLUSTER_DEGRADED')
            demand(record.get('smoke_passed') is True, 24, 'SMOKE_NOT_VERIFIED')
            api_identity(config, time.monotonic()+3)
    print(json.dumps({'phase': phase, **config.identity}))
    return 0 if phase == 'ready' else 24


def stop(config):
    record = config.read_state()
    if not record:
        demand(not lock_held(config.state / 'instance.lock'), 22, 'INSTANCE_STARTING_RETRY_STOP')
        print('STOPPED'); return 0
    config.matching_state(record)
    if record['phase'] not in ACTIVE_PHASES:
        ensure_no_live_record(record)
        print('STOPPED'); return 0
    try:
        demand(lock_held(config.state / 'instance.lock'), 22, 'STALE_OR_REUSED_PID_REFUSED')
        # Signal 0 (pidfd on Linux) is only an ownership/liveness probe. The
        # run-ID request is safe under concurrent stop, owner exit and PID reuse.
        signal_identity(record.get('owner'), 0)
        atomic_json(config.state / 'stop-request.json', {'run_id': record['run_id']})
    except RuntimeFailure:
        current = config.read_state()
        if (current and current.get('run_id') == record['run_id'] and current['phase'] in ('stopped', 'failed')
                and not current.get('owner') and not current.get('child') and not current.get('cluster_active')):
            print('STOPPED'); return 0
        raise
    deadline = time.monotonic()+STOP_WAIT_SECONDS
    while time.monotonic() < deadline:
        current = config.read_state()
        if current and current.get('run_id') != record['run_id']:
            raise RuntimeFailure(22, 'INSTANCE_CHANGED_DURING_STOP')
        if current and current['phase'] in ('stopped', 'failed') and not current.get('owner') and not current.get('cluster_active'):
            print('STOPPED'); return 0
        time.sleep(.05)
    raise RuntimeFailure(24, 'STOP_TIMEOUT: inspect locally; no unsafe external escalation attempted')


def logs(config, lines=80, follow=False, rank=0):
    record = config.read_state()
    if record:
        config.matching_state(record)
    if config.kind == 'cluster':
        demand(not follow, 2, 'CLUSTER_LOG_FOLLOW_UNSUPPORTED: use repeated --no-follow')
        result = config.cluster_call('logs', ('--rank', str(rank), '--lines', str(lines)))
        demand(result.returncode == 0, 24, 'CLUSTER_LOGS_FAILED')
        print(result.stdout, end=''); return 0
    demand(record is not None and re.fullmatch(r'[0-9a-f]{32}', record.get('run_id', '')) is not None, 24, 'NO_OWNED_LOG')
    path = safe_file(config.state, 'logs/' + record['run_id'] + '.log')
    with path.open(encoding='utf-8', errors='replace') as stream:
        print(''.join(deque(stream, maxlen=lines)), end='', flush=True)
        while follow:
            line = stream.readline()
            if line:
                print(line, end='', flush=True)
            else:
                if not lock_held(config.state / 'instance.lock'):
                    break
                time.sleep(.1)
    return 0


def diagnostics(config):
    record = config.read_state() or {}
    report = {'schema': 'rexl3.diagnostics.v1', 'runtime_id': config.profile['runtime_id'], **config.identity,
              'phase': record.get('phase', 'stopped'), 'state_matches': all(record.get(k) == v for k, v in config.identity.items()),
              'bundle_integrity': 'verified', 'model_revision': config.models['packs']['target']['revision'],
              'production_commit': config.release['production_commit'],
              'server_exit_code': record.get('server_exit_code') if type(record.get('server_exit_code')) is int else None}
    secure_directory(config.state)
    atomic_json(config.state / 'diagnostics.json', report)
    print(json.dumps(report, indent=2)); return 0


def reset_state(config):
    secure_directory(config.state)
    with advisory_lock(config.state / 'instance.lock'):
        ensure_no_live_record(config.read_state())
        for name in ('instance.json', 'stop-request.json'):
            path = config.state / name
            if path.exists():
                demand(not path.is_symlink(), 22, 'UNSAFE_INSTANCE_STATE')
                path.unlink()
    print('STATE_RESET: models, logs, dependency environment and user configuration preserved')
    return 0


def main(argv=None, root=ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', default='start', choices=('setup', 'print-command', 'start', 'status', 'stop', 'logs', 'verify', 'diagnostics', 'reset-state'))
    parser.add_argument('--profile')
    parser.add_argument('--site', type=Path)
    parser.add_argument('--accept-model-download', action='store_true')
    follow = parser.add_mutually_exclusive_group()
    follow.add_argument('--follow', action='store_true')
    follow.add_argument('--no-follow', action='store_true')
    parser.add_argument('--lines', type=int, default=80)
    parser.add_argument('--rank', type=int, default=0)
    options = parser.parse_args(argv)
    try:
        demand(1 <= options.lines <= 1000 and options.rank >= 0, 2, 'INVALID_LOG_OPTIONS')
        config = Configuration(root, options.profile, options.site)
        command = options.command
        if command == 'print-command':
            print(json.dumps(config.output(), indent=2)); return 0
        if command == 'start':
            return start(config, options.accept_model_download)
        if command == 'setup':
            import doctor
            result = doctor.main([], root=config.root)
            demand(result == 0, result or 20, 'PREFLIGHT_FAILED')
            prepare_models(config.root, accept_download=options.accept_model_download)
            print('SETUP_VERIFIED'); return 0
        if command == 'verify':
            prepare_models(config.root, verify_only=True)
            print('BUNDLE_AND_MODELS_VERIFIED'); return 0
        if command == 'status':
            return status(config)
        if command == 'stop':
            return stop(config)
        if command == 'logs':
            return logs(config, options.lines, options.follow, options.rank)
        if command == 'diagnostics':
            return diagnostics(config)
        return reset_state(config)
    except RuntimeFailure as error:
        print(str(error), file=sys.stderr); return error.code
    except ContractError as error:
        # Even invalid diagnostics inputs must not echo arbitrary duplicate keys,
        # file paths or other user-controlled strings into a shareable report.
        print('DIAGNOSTICS_REFUSED_INVALID_CONTRACT' if options.command == 'diagnostics'
              else 'CONTRACT_INVALID: ' + str(error), file=sys.stderr)
        integrity = ('BUNDLE_', 'CONFIGURATION_HASH_', 'SYMLINK_', 'UNSAFE_OR_MISSING_FILE', 'RELEASE_INVENTORY_', 'EXTRA_MANAGED_')
        return 23 if str(error).startswith(integrity) else 2
    except (OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError):
        print('RUNTIME_OPERATION_FAILED: inspect the owned log locally', file=sys.stderr); return 24
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
