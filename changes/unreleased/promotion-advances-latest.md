## Bug fixes

- `tools/promote_runtime.py` can now promote a later version of a model that is already on `main`. It used to refuse with `existing LATEST differs` whenever `LATEST` named an earlier version, so no requalified version could be promoted. It now moves `LATEST` forward from `v<M>` to the new `v<N>`, and still refuses a `LATEST` that names a later version, is malformed or is not a regular file. PR #137.
