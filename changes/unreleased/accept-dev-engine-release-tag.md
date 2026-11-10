## Bug fixes

- Production verification and the release manifest now accept the engine release tag of a development engine build, `engine-v<version>-dev-<commit8>` (for example `engine-v0.1.1-dev-2067531c`). The engine reports its version as `0.1.1-dev` on the development line, so the tag that names it carries `-dev`; the old pattern refused it. The tag must still equal the native engine version and the first eight hex digits of its commit, and nothing else about the tag check is loosened. PR #117.
