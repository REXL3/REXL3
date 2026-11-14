"""Where a new approval anchor may come from: a release PR of this repository that lists it.

`verify` proves that a newly anchored tree equals a published dev seal byte for byte. That proves
the bytes are a real seal; it does not prove the maintainers chose to release that seal. The choice
is the maintainers' review of a `release/v<X.Y.Z>` pull request, so a new anchor is accepted only
when the pull request's head branch is `release/v<X.Y.Z>` in the base repository itself (not a fork)
and `releases/v<X.Y.Z>/manifest.json` in the candidate lists that anchor with the same approval hash.

The head repository and head ref are untrusted strings from the pull-request event. They are only
compared and matched here, never echoed, executed or used to build a path before they match.
"""
from pathlib import Path
import re
try:
    from . import release_evidence as e
    from . import model_layout as m
    from .release_manifest import validate_manifest
except ImportError:
    import release_evidence as e
    import model_layout as m
    from release_manifest import validate_manifest

REPOSITORY = re.compile(r'[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9._-]+')
RELEASE_HEAD = re.compile(r'release/(v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))')


def release_head_version(base_repo, head_repo, head_ref):
    """`v<X.Y.Z>` when the head is `release/v<X.Y.Z>` in the base repository; refuse anything else."""
    e.require(isinstance(base_repo, str) and REPOSITORY.fullmatch(base_repo) is not None,
              'new approval anchors need the pull request context (base repository, head repository, head ref)')
    e.require(isinstance(head_repo, str) and head_repo == base_repo,
              'new approval anchors are accepted only from a branch of this repository, not from a fork')
    match = RELEASE_HEAD.fullmatch(head_ref) if isinstance(head_ref, str) else None
    e.require(match is not None, 'new approval anchors are accepted only from a release/v<X.Y.Z> head branch')
    return match[1]


def require_release_pr(new_keys, candidate_registry, *, root, base, base_repo=None, head_repo=None, head_ref=None):
    """Refuse new anchors that do not arrive in a release PR whose manifest lists them.

    ``new_keys`` are the registry keys the candidate adds to the protected base. With none, nothing
    is required (every other pull request, and every push to main). Returns the release version.
    """
    if not new_keys:
        return None
    version = release_head_version(base_repo, head_repo, head_ref)
    # A released version's manifest is already on the base; its branch name cannot be used again.
    e.require(not (Path(base) / 'releases' / version).exists(),
              'release already has files on the protected base: ' + version)
    manifest = validate_manifest(e.load_json(e.safe_file(root, f'releases/{version}/manifest.json')), version)
    listed = {m.registry_key(item['family'], item['model'], int(item['version'][1:])): item
              for item in manifest['models']}
    approvals = candidate_registry.get('approvals', {})
    for key in sorted(new_keys):
        item, entry = listed.get(key), approvals.get(key)
        e.require(item is not None, 'new approval anchor is not listed in the release manifest: ' + key)
        e.require(isinstance(entry, dict) and item['approval_sha256'] == entry.get('approval_sha256')
                  and item['runtime_id'] == entry.get('runtime_id'),
                  'release manifest does not match the new approval anchor: ' + key)
    return version


def add_arguments(parser):
    """The pull-request context options shared by fetch_dev_assets.py and verify_runtime_tree.py."""
    parser.add_argument('--base-repo', help='OWNER/REPO the pull request targets')
    parser.add_argument('--head-repo', help='OWNER/REPO of the pull request head (untrusted)')
    parser.add_argument('--head-ref', help='head branch of the pull request (untrusted)')


def context(args):
    return {'base_repo': args.base_repo or None, 'head_repo': args.head_repo or None, 'head_ref': args.head_ref or None}
