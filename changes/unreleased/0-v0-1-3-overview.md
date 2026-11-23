## Highlights

- v0.1.3 is a fix release. Every runtime choice of v0.1.1 and v0.1.2 was qualified again, with a fresh full gate chain, on one engine build: engine 0.1.3, engine commit `0c8bc43c`, release binary `rexl3-serve` sha256 `bd77b147…`. No gate result is carried from an earlier engine.
- It ships the fixes for the tester reports listed under Bug fixes: driver and NCCL updates no longer stop an install, the two-node DeepSeek-V4.1 rank check passes, fresh DeepSeek-V4.1 downloads work again, and launcher stops are clean.
- Ling-3.1-Flash-1019 now thinks by default, so a request that names no thinking setting gets a clean answer.
- The known issues listed below replace the "None" of the v0.1.2 notes.
- Every qualification chain ran on the same published release binary, and each seal is verified again by a second maintainer before publication.
