## Known issues

- Ling-3.1-Flash-1019 is not in v0.1.0; it moves to v0.1.1.
- DeepSeek-V4.1-Flash-1017's `tp2` two-node profile is not in v0.1.0; only its `tp1` single-node
  profile (1.55 bpw) ships. The earlier prefill chunk-invariance issue on the v0.1.0 build is
  resolved: the `tp1` seal uses #229's V57 chunk-invariant attention (D37), and `tp2` ships as a
  separate runtime unit (D38).
