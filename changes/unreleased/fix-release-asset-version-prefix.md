## Bug fixes

- Release assets are named `<runtime-id>-v<N>-approval.json` and `<runtime-id>-v<N>-bundle.tar.gz`
  (plus the `-source`, `-engine-source` and `-evidence` archives) as documented. The release tool
  had written the model version's `v` twice (`<runtime-id>-vv1-...`), and the changelog's model line
  had the same doubled `v`. (PR #49)
