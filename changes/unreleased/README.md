# Release-notes fragments

A fragment is one small markdown file under `changes/unreleased/` describing one change for the
release notes. Fragments are consumed at release time by `tools/assemble_release_notes.py` into the
template `docs/templates/RELEASE_NOTES.md` (docs/RELEASING.md).

## When a fragment is required

- `model/*` PRs: always — `changes/unreleased/<model>.md`.
- `engine/*` PRs that change behavior (tools, workflows): yes — `changes/unreleased/<slug>.md`.
- DOCS-only PRs: no.

## Rules

- Filename: lowercase kebab-case `<slug>.md` — for example `qwen3.8-27b.md`,
  `fix-tag-protection.md`. One change per fragment.
- Content: markdown using only the headings it needs from the release-notes template: Highlights;
  Supported models; New features and improvements; Bug fixes; Breaking changes and upgrade notes;
  Known issues; Assets and checksums; Provenance.
- A few sentences; say what changed and why it matters to someone installing the release.
- Mention the PR number. No secrets, tokens, personal email addresses or internal project names.
- Do not edit a fragment that belongs to another PR.
