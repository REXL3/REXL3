## Highlights

- **GLM-5.3-Flash** (EXL3 2.25 bits per weight, SAGE) on two GB10 nodes, version 2 (`glm53-flash-tp2-1m`: tensor parallel, the full 1,048,576-token window, 1 slot, port 8950, speculation, prefix cache and image input on), requalified on engine 0.1.3 with a fresh full gate chain.
  - Gates: all passed. Exactness 68 of 68, including chunked prefill equal to whole prefill on a 1,047,682-token prompt. Context needles 16 of 16 at 131,072 to 1,048,576 tokens. An extra full-window check also passed: drafted output equal to serial output, and cached equal to cold, at 1,048,576 tokens.
  - Speed on the release binary (two GB10s): 56.6 tok/s generation for one stream (median of 3 runs, each the median of 5 replies of 256 tokens, short context). Prompt speed 1,139 tok/s at 50% and 985 tok/s at 100% of the window (cold prompts, median of 3 runs).
  - Q200 184 of 200 (92.0%) at reasoning effort `xhigh`, greedy, with a 65,536-token text budget and 8,192 for BFCL. One client worker ran remote from the leader. 3 text replies ran to the budget and count as incorrect. GSM8K 77/80, HumanEval 39/40, IFEval 37/40, hard reasoning 19/20, BFCL hard-20 12/20. Every suite is at or above the floors sealed for version 1.
  - v0.1.1 measured 56.8 tok/s and 1,133 / 980 tok/s, with the same Q200 total.
