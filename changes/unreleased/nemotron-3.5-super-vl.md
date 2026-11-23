## Highlights

- **Nemotron-3.5-Super-VL** (EXL3 3.85 bits per weight, SAGE), version 3: the same pack and single-node profile (`nemotron35-super-vl-single-1m`: one GB10, 1,048,576 tokens, 16 slots, port 9100, speculation, prefix cache and vision on), requalified on engine 0.1.3 with a fresh full gate chain. Context and quality, which version 2 carried from version 1, were measured again.
  - Gates: all passed. Exactness 68 of 68, including chunked prefill equal to whole prefill on a prompt near the full window. Context needles 16 of 16 at 131,072 to 1,048,576 tokens.
  - Speed on the release binary: 50.0 tok/s generation for one stream (median of 3 runs, each the median of 5 streamed replies of 256 tokens, short context). Prompt speed 861 and 575 tok/s at 50% and 100% of the window (cold prompts, median of 3 runs). Lowest free memory 26.9 GiB, with no swap.
  - Q200 179 of 200 (89.5%) at reasoning effort `xhigh`, greedy, a 65,536-token text budget and 8,192 for BFCL, one client worker on another host; 8 text replies ran to the budget and count as incorrect. GSM8K 76/80, HumanEval 40/40, IFEval 28/40, hard reasoning 20/20, BFCL hard-20 15/20, every suite at or above the floors sealed for version 2.
  - v0.1.1 measured 50.1 tok/s and 881 / 589 tok/s.
