## Highlights

- **Qwen3.8-27B** (EXL3 4.00 bits per weight with a DFlash2 draft pack), version 5: the same packs and single-node profile (`qwen38-single-262k`: one GB10, 262,144 tokens, 6 slots, port 8889, speculation and prefix cache on, vision on), requalified on engine 0.1.3 with a fresh full gate chain.
  - Gates: all passed. Exactness 68 of 68, including chunked prefill equal to whole prefill at five prompt lengths up to 261,120 tokens, with 256- and 512-token chunks. Context needles 8 of 8 at 131,072 and 262,144 tokens.
  - Speed on the release binary: 62.5 tok/s generation for one stream (median of 3 runs, each the median of 5 streamed replies of 256 tokens, short context). Prompt speed 839 and 638 tok/s at 50% and 100% of the window (cold prompts, median of 3 runs). No swap during those runs.
  - Q200 188 of 200 (94.0%) at reasoning effort `low`, greedy, a 32,768-token text budget and 8,192 for BFCL, one client worker on the same host: GSM8K 79/80, HumanEval 39/40, IFEval 35/40, hard reasoning 20/20, BFCL hard-20 15/20. The floor is the sealed 70% in every suite. v0.1.1 measured 62.2 tok/s and 836 / 635 tok/s.
