## Bug fixes

- Made the governance audit tolerate GitHub Actions workflow-token limits: a 403 on the
  immutable-releases settings endpoint and incomplete/403 ruleset detail views are skipped with an
  informational line instead of failing the audit (real, readable drift still fails). The immutable
  releases setting remains verifiable by a maintainer token.
- The incident record now enumerates the full retroactive review range (`#1-#28`), and the audit
  parser accepts `through`-spelled ranges, so pre-restructure unreviewed merges are matched.
