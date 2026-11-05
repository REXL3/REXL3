## Known issues

- Nemotron-3.5-Super-VL accepts images of up to 3,328 vision tokens each and refuses audio.
- Nemotron-3.5-Super-VL takes video only as a list of image frames (at most 32 frames or 4,096 tokens); video files (`data:video/mp4`, `webm`, `gif`) are refused. Video sent as frames works but is not covered by the v0.1.0 qualification.
