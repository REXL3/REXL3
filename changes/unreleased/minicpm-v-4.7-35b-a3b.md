## Highlights

- **MiniCPM-V-4.7-35B-A3B** (EXL3 6.08 bits per weight, SAGE, with a DSpark draft pack), version 2: the same packs and single-node profile (`minicpmv47-single-262k`: one GB10, 262,144 tokens, 4 slots, port 8889, speculation, prefix cache and image input on), requalified on engine 0.1.3 with a fresh full gate chain.
  - Gates: all passed. Exactness 68 of 68, including chunked prefill equal to whole prefill up to a 261,120-token prompt. Context needles 8 of 8.
  - Speed on the release binary: 111.8 tok/s generation for one stream (median of 3 runs, each the median of 5 streamed replies of 256 tokens, draft on, short context). Prompt speed 2,669 and 1,848 tok/s at 50% and 100% of the window (cold prompts, median of 3 runs). Lowest free memory 41.3 GiB, with no swap.
  - Q200 183 of 200 (91.5%) with thinking on, greedy, an 8,192-token budget, a client on the same host sending the text suites two requests at a time: GSM8K 76/80, HumanEval 40/40, IFEval 34/40, hard reasoning 20/20, BFCL hard-20 13/20. Every suite is at or above the floors sealed for version 1. v0.1.1 measured 112.2 tok/s and 2,706 / 1,884 tok/s.
