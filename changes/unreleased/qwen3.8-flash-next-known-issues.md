## Known issues

- Qwen3.8-Flash-Next's served model name is `default`, not a name derived from the model. It is the name `GET /v1/models` lists and the one requests send as `model`.
- Qwen3.8-Flash-Next runs with vision off, under an owner vision exception that expires on 2027-01-31: the engine does not build this model's vision tower yet. Image input is refused.
- Qwen3.8-Flash-Next drafts with MTP only while one request is decoding; when both slots are decoding, generation runs without drafts. Replies are the same either way.
