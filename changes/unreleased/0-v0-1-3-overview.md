## Highlights

- v0.1.3 is a fix release. Every runtime choice of v0.1.1 and v0.1.2 was qualified again, with a fresh full gate chain, on one engine build: engine 0.1.3, dev commit `0c8bc43c` (`release/v0.1.3`). Each node built the same `rexl3-serve` bytes (sha256 `bd77b147…`) independently from that commit. No gate result is carried from an earlier engine.
- It ships the fixes for the tester reports listed under Bug fixes: driver and NCCL updates no longer stop an install, the two-node DeepSeek-V4.1 rank check passes, fresh DeepSeek-V4.1 downloads work again, and launcher stops are clean.
- Ling-3.1-Flash-1019 now thinks by default, so a request that names no thinking setting gets a clean answer.
- The known issues listed below replace the "None" of the v0.1.2 notes.
- The maintainers split the qualification chains for this release across both sides' GB10 nodes. Some choices that @vcruz305 maintains (both Qwen3.8-Flash-Next choices) ran on the other maintainer's nodes, and the rest of his choices ran on his own. Every chain used the same published release binary, and each side's seals are re-verified by the other before publication.
