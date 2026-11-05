#!/usr/bin/env python3
"""Assemble release notes from `changes/unreleased/<slug>.md` fragments (contract section 7).

    python3 tools/assemble_release_notes.py --repo-root PATH --version vX.Y.Z

Fragments use only the headings they need; each heading maps to a template section. A fragment is a
`<slug>.md` file with a lowercase kebab-case slug (changes/unreleased/README.md); that README itself holds
the fragment rules and is not a fragment. The Supported models table, Assets and checksums, and
Provenance are generated from
releases/v<X.Y.Z>/manifest.json (schema rexl3.release-manifest.v1). Writes
releases/v<X.Y.Z>/NOTES.md and prepends the CHANGELOG.md entry. Never edits anything else.
"""
import argparse
from datetime import date
import os
from pathlib import Path
import re
import sys

try:
    from . import release_evidence as e
    from . import model_layout as m
    from .release_manifest import validate_manifest
except ImportError:
    import release_evidence as e
    import model_layout as m
    from release_manifest import validate_manifest

FRAGMENT_HEADINGS = {
    'Highlights': 'HIGHLIGHTS',
    'New features and improvements': 'NEW_FEATURES',
    'Bug fixes': 'BUG_FIXES',
    'Breaking changes and upgrade notes': 'BREAKING_CHANGES',
    'Known issues': 'KNOWN_ISSUES',
}
# changes/unreleased/README.md documents the fragment rules; it is the one file there that is not a fragment.
RULES_FILE = 'README.md'
# A fragment's filename is `<slug>.md`, the slug lowercase kebab-case (dots allowed, as in `qwen3.8-27b.md`).
FRAGMENT_SLUG_RE = re.compile(r'[a-z0-9]+(?:[.-][a-z0-9]+)*')


def parse_fragments(fragments_dir):
    """Return {placeholder: [lines]} from all *.md fragments (not the rules README).

    A file whose name is not a fragment slug, an empty fragment or a duplicate slug fails.
    """
    collected = {placeholder: [] for placeholder in FRAGMENT_HEADINGS.values()}
    if not fragments_dir.is_dir():
        raise ValueError('fragments directory missing: ' + str(fragments_dir))
    fragments = sorted(path for path in fragments_dir.glob('*.md') if path.name != RULES_FILE)
    for path in fragments:
        if not FRAGMENT_SLUG_RE.fullmatch(path.stem):
            raise ValueError(path.name + ': fragment name is not a lowercase kebab-case <slug>.md')
    if not fragments:
        raise ValueError('no fragments in ' + str(fragments_dir))
    seen = {}
    for path in fragments:
        slug = path.stem
        if slug in seen:
            raise ValueError('duplicate fragment slug: ' + slug)
        seen[slug] = True
        text = path.read_text(encoding='utf-8')
        if not text.strip():
            raise ValueError('empty fragment: ' + path.name)
        lines = text.splitlines()
        current = None
        for line in lines:
            heading = re.fullmatch(r'##\s+(.+)', line)
            if heading:
                title = heading.group(1).strip()
                if title not in FRAGMENT_HEADINGS:
                    raise ValueError(f'{path.name}: unknown heading {title!r}')
                current = FRAGMENT_HEADINGS[title]
                continue
            if current is None:
                continue  # prose before the first heading is ignored
            collected[current].append(line)
        if current is None:
            raise ValueError(path.name + ': fragment has no recognized ## heading')
    return collected


def supported_models_table(manifest):
    rows = ['| Model | Architecture | Pack | Context | Speculative decoding | Prefix cache | Vision | Evidence |',
            '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for item in manifest['models']:
        rows.append(f"| {item['family']}/{item['model']} | {item.get('architectures','')} | "
                    f"{item.get('pack_format','')} | {item.get('context','')} | "
                    f"{item.get('speculative_decoding','')} | {item.get('prefix_cache','')} | "
                    f"{item.get('vision','')} | {item.get('evidence','')} |")
    return '\n'.join(rows)


def assets_section(manifest):
    lines = []
    for name, digest in manifest['engine']['artifacts'].items():
        lines.append(f'- `{name}` sha256 `{digest}`')
    for item in manifest['models']:
        lines.append(f"- {item['family']}/{item['model']} {item['version']}: approval sha256 `{item['approval_sha256']}`")
    return '\n'.join(lines) or 'None.'


def provenance_section(manifest, repo_root=None):
    engine = manifest['engine']
    lines = [f'- Engine dev commit `{engine["dev_commit"]}` tree `{engine["dev_tree"]}`; '
             f'dev engine release tag `{engine["engine_release_tag"]}`.']
    root = Path(repo_root) if repo_root is not None else None
    fresh, carried = [], []
    for item in manifest['models']:
        if item.get('carried_forward') is True:
            commit = None
            if root is not None:
                approval_path = (root / 'runtimes' / item['family'] / item['model']
                                / item['version'] / 'approval.json')
                if approval_path.is_file():
                    commit = e.load_json(approval_path).get('source_commit')
            e.require(isinstance(commit, str) and e.HEX40.fullmatch(commit),
                      'carried-forward approval source commit unavailable for provenance: '
                      + f"{item['family']}/{item['model']} {item['version']}")
            carried.append((item, commit))
        else:
            fresh.append(item)
    if not carried:
        lines.append('- All models are sealed from the same dev engine release commit.')
        return '\n'.join(lines)
    if fresh:
        lines.append('- Models sealed from the engine dev commit `%s`: %s.'
                     % (engine['dev_commit'],
                        ', '.join(f"{i['family']}/{i['model']} {i['version']}" for i in fresh)))
    else:
        lines.append(f'- No model is sealed from the engine dev commit `{engine["dev_commit"]}`; '
                     'every model carries forward its previously sealed approval.')
    for item, commit in carried:
        lines.append(f"- {item['family']}/{item['model']} {item['version']} carries forward its "
                     f"previously sealed approval at commit `{commit}` (engine release "
                     f"`{item['qualified_engine_release_tag']}`).")
    return '\n'.join(lines)


def render(template_text, version, manifest, fragments, repo_root=None):
    for placeholder in FRAGMENT_HEADINGS.values():
        body = '\n'.join(fragments[placeholder]).strip()
        template_text = template_text.replace('{{' + placeholder + '}}', body or 'None.')
    template_text = template_text.replace('{{VERSION}}', version)
    template_text = template_text.replace('{{SUPPORTED_MODELS}}', supported_models_table(manifest))
    template_text = template_text.replace('{{ASSETS}}', assets_section(manifest))
    template_text = template_text.replace('{{PROVENANCE}}', provenance_section(manifest, repo_root))
    if '{{' in template_text:
        raise ValueError('unresolved template placeholder')
    return template_text


def changelog_entry(version, manifest, fragments):
    lines = [f'## [{version}] - {date.today().isoformat()}', '']
    highlights = '\n'.join(fragments['HIGHLIGHTS']).strip()
    if highlights:
        lines.append(highlights)
        lines.append('')
    multi = [line for placeholder in ('NEW_FEATURES', 'BUG_FIXES') for line in fragments[placeholder]]
    if multi:
        lines.extend(multi)
        lines.append('')
    lines.append(f'- Models: ' + ', '.join(f"{x['family']}/{x['model']} {x['version']}" for x in manifest['models']) + '.')
    return '\n'.join(lines)


def assemble(repo_root, version, *, fragments_dir=None, template=None, notes=None, changelog=None,
            manifest=None):
    m.product_version(version)
    root = Path(repo_root)
    fragments_dir = Path(fragments_dir) if fragments_dir else root / 'changes/unreleased'
    template = Path(template) if template else root / 'docs/templates/RELEASE_NOTES.md'
    notes = Path(notes) if notes else root / 'releases' / version / 'NOTES.md'
    changelog = Path(changelog) if changelog else root / 'CHANGELOG.md'
    manifest = Path(manifest) if manifest else root / 'releases' / version / 'manifest.json'
    if not template.is_file():
        raise ValueError('release notes template missing: ' + str(template))
    doc = validate_manifest(e.load_json(manifest), version)
    fragments = parse_fragments(fragments_dir)
    rendered = render(template.read_text(encoding='utf-8'), version, doc, fragments, repo_root=root)
    notes.parent.mkdir(parents=True, exist_ok=True)
    if notes.exists():
        e.require(notes.read_text(encoding='utf-8') == rendered, 'existing NOTES.md differs')
    else:
        notes.write_text(rendered, encoding='utf-8')
    entry = changelog_entry(version, doc, fragments) + '\n'
    if changelog.exists():
        text = changelog.read_text(encoding='utf-8')
        header = '# Changelog\n\n' if not text.startswith('# Changelog') else '# Changelog\n\n'
        body = text[len(header):] if text.startswith('# Changelog') else text
        if f'[{version}]' in text:
            existing_entry = text.split(f'[{version}]', 1)[1]
            new_entry = (header + entry).split(f'[{version}]', 1)[1]
            e.require(existing_entry == new_entry, f'CHANGELOG.md has a different entry for {version}')
            return 0
        changelog.write_text(header + entry + '\n' + body, encoding='utf-8')
    else:
        changelog.write_text('# Changelog\n\n' + entry, encoding='utf-8')
    print('WROTE ' + str(notes))
    print('WROTE ' + str(changelog))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', default='.')
    parser.add_argument('--version', required=True)
    args = parser.parse_args(argv)
    try:
        return assemble(args.repo_root, args.version)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print('NOTES_REFUSED: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
