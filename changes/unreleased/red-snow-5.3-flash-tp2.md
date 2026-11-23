## Highlights

- **RED-SNOW-5.3-Flash** (EXL3 4.91 bits per weight, SAGE; a GLM-5.3-Flash derivative) on two GB10 nodes, version 2 (`redsnow-tp2-512k`: tensor parallel, 524,288-token window, 1 slot, port 8952, speculation with the pack's MTP layer, pool-page prefix cache and image input on), requalified on engine 0.1.3 with a fresh full gate chain. Context and quality, which version 1 carried from the previous engine build under a bit-identity record, were measured again.
  - Gates: all passed. Exactness 68 of 68, including chunked prefill equal to whole prefill on a 522,912-token prompt. Context needles 12 of 12 at 131,072, 262,144 and 524,288 tokens.
  - Speed on the release binary (two GB10s): 34.4 tok/s generation for one stream (median of 3 runs, each the median of 5 replies of 256 tokens, MTP on, short context). Prompt speed 1,103 tok/s at 50% and 1,017 tok/s at 100% of the window (cold prompts, median of 3 runs).
  - Q200 161 of 200 (80.5%) with thinking off, greedy, an 8,192-token budget for text and BFCL. One client worker ran remote from the leader, and no text reply ran to the budget. GSM8K 69/80, HumanEval 39/40, IFEval 30/40, hard reasoning 15/20, BFCL hard-20 8/20. Every suite is at or above the floors sealed for version 1, and the total is above its 80% bar.
  - v0.1.1 measured 34.5 tok/s and 1,097 / 1,014 tok/s, with Q200 160 of 200.
