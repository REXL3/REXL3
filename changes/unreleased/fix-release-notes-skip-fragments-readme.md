## Bug fixes

- The release-notes assembler (`tools/assemble_release_notes.py`) no longer reads
  `changes/unreleased/README.md`, the fragment rules, as a fragment. Its own headings had made every
  assembly on main fail with `NOTES_REFUSED: README.md: unknown heading 'When a fragment is required'`,
  so the documented release step could not run. A file there whose name is not a lowercase kebab-case
  `<slug>.md` is now refused by name instead of being parsed.
