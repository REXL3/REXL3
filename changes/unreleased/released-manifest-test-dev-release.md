## Bug fixes

- The release-manifest test suite passes again on main. It had required every released manifest to publish its evidence archives, but v0.1.3 keeps them on the development release with `"evidence_assets": "dev-release"`. The test now expects publication exactly when that switch is absent (#197).
