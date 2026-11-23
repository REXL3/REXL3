## Highlights

- **Qwen3.8-Flash-Next** (EXL3 3.05 bits per weight), version 4, one GB10, 262,144 tokens, 2 slots, port 8899, prefix cache on, image input off under the existing exception. Both choices were requalified on engine 0.1.3, each with its own fresh full gate chain and Q200 run.
  - **Default choice `qwen38-flash-next-single-262k` (MTP speculation).** All gates passed: exactness 68 of 68, including chunked prefill equal to whole prefill up to a 261,120-token prompt, and context needles 8 of 8.
    - Speed on the release binary: 71.7 tok/s generation for one stream (median of 3 runs, each the median of 5 streamed replies of 256 tokens, short context). Prompt speed 1,135 and 1,061 tok/s at 50% and 100% of the window (cold prompts, median of 3 runs). Lowest free memory 26.8 GiB, with no swap.
  - **Opt-in choice `qwen38-flash-next-single-262k-draft` (DFlash2 draft pack `r0b0tlab/Qwen3.8-Flash-Next-DFlash2-EXL3-4.00bpw`).** All gates passed: exactness 68 of 68 and context needles 8 of 8.
    - Speed by the same method: 69.5 tok/s for one stream, and 1,166 / 1,098 tok/s prompt speed. Lowest free memory 30.2 GiB, with no swap.
  - Q200, each choice at reasoning effort `xhigh`, greedy, a 65,536-token text budget and 8,192 for BFCL, one client worker on the same host: 185 of 200 (92.5%) for both. GSM8K 78/80, HumanEval 40/40, IFEval 38/40, hard reasoning 20/20, BFCL hard-20 9/20. That is every suite at or above the floors sealed for version 3; 1 text reply ran to the budget and counts as incorrect.
  - The decode, prefill and graph-cache changes listed under New features and Bug fixes are in this version. v0.1.1 measured 69.5 tok/s and 1,047 / 996 tok/s for the default choice, and 63.0 tok/s and 1,027 / 979 tok/s for the drafting choice.
