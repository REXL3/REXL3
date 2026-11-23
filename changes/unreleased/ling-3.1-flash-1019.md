## Highlights

- **Ling-3.1-Flash-1019** (EXL3 1.51 bits per weight, SAGE), version 2: the same pack and single-node profile (`ling31-single-262k`: one GB10, 262,144 tokens, 1 slot, port 8889, speculative decoding with the pack's MTP head, prefix cache on, text only), requalified on engine 0.1.3 with a fresh full gate chain. It now thinks by default and plans its boot from exact free memory (see the upgrade notes).
  - Gates: all passed. Exactness 68 of 68, including chunked prefill equal to whole prefill up to a 261,120-token prompt. Context needles 8 of 8 at 131,072 and 262,144 tokens.
  - Speed on the release binary: 30.8 tok/s generation for one stream (median of 3 runs, each the median of 5 streamed replies of 256 tokens, short context). Prompt speed 325 and 217 tok/s at 50% and 100% of the window (cold prompts, median of 3 runs).
  - Memory during those runs: lowest free memory 2.3 GiB, and up to 0.12 GiB of swap used. Start this profile on an idle host.
  - Q200 179 of 200 (89.5%) with thinking on, a 32,768-token budget, one client worker on another host; 8 text replies ran to the budget and count as incorrect. GSM8K 75/80, HumanEval 40/40, IFEval 32/40, hard reasoning 20/20, BFCL hard-20 12/20, every suite at or above the floors sealed for version 1.
  - The Q200 summary was rebuilt from the run's stored rows with a later revision of the scoring tool, which counts an answer cut off at the budget as incorrect, as for every other runtime. The engine, the binary and the converter are those of the v0.1.3 engine commit `0c8bc43c`.
  - v0.1.1 measured 27.1 tok/s and 341.8 / 223.3 tok/s.
