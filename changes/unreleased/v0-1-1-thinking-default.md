## Known issues

- Qwen3.8-Flash-Next and Nemotron-3.5-Super-VL: a request that names no reasoning setting does not think, although each pack's chat template would; some earlier documentation said thinking was on by default. To turn it on, send it in `chat_template_kwargs`, for example `{"chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "xhigh"}}`, which is what their qualification runs sent, and leave room in `max_tokens` for the thinking (the quality runs used 65,536 tokens). Their pack sheets and the user guide (section 5.3) say the same. Added by #149.
