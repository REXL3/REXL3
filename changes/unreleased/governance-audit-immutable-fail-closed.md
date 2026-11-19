## Bug fixes

- Fixed the governance audit's immutable-releases check to fail closed when the settings response has no boolean `enabled` field (for example `{}` or `"enabled": null`), as its documentation says. Previously only an explicit `"enabled": false` was reported. Covered by a new unit test.
