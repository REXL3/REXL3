## Bug fixes

- `tools/cut_release.py` now refuses a cut unless the repository's immutable-releases setting answers
  with `enabled` set to the boolean `true`. A response without `enabled`, or with `null`, `false` or a
  non-boolean value, used to pass the gate; it now stops the dry run and `--publish` before any tag or
  release is written. Reported by @Blackwellboy in #158 (#162).
