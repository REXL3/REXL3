## Known issues

- Served model names (the `model` field of a request, as `GET /v1/models` lists it) differ in style from model to model in v0.1.0: Qwen3.8-27B `qwen38-27b-exl3`, Qwen3.8-Flash-Next `default`, GLM-5.3-Flash `GLM-5.3-Flash-EXL3-2.25bpw-sm121`, Nemotron-3.5-Super-VL `nemotron-3.5-super-vl`. Read the name from `GET /v1/models` or the recipe's `profile.json` instead of deriving it.
