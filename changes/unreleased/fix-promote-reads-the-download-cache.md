## Bug fixes

- `tools/promote_runtime.py` can now promote a new model with its documented command. Two defects stopped it.
  `--download` caches a dev release in `<assets-dir>/<family>/<model>/v<N>/`, but the promotion read
  `approval.json`, the archives and the engine executables from `<assets-dir>` itself and refused with
  "missing/nonregular/hardlinked file: approval.json"; it now reads that model version's cache directory, the
  same one the tree check uses. And the early check of the `MODEL.json` metadata flags left out the runtime id,
  so every promotion that passed `--display-name` and the other metadata flags, which a new model needs, was
  refused before it started; the check now uses the registry entry's runtime id.
