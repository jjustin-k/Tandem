# Local cache replay results

Measured on a 16 GB M2 MacBook Air, 27 October 2026. Model: Qwen2.5-0.5B-Instruct at `7ae557604adf67be50417f59c2c2f167def9a775`, FP32, eager attention, MPS. Nine cases: notes of 128, 512, and 1,024 tokens, with the planted code early, middle, or late. Each time below is the median of nine first-token measurements (three positions, three repetitions).

Speedup is the native time divided by that arm's time. Native reads the notes and the question. Warm reuses an in-memory cache and reads only the question. Restored loads the cache from a SafeTensors file, checks it, and reads only the question.

| Notes | Native | Warm | Warm speedup | Restored | Restored speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 tokens | 82.55 ms | 43.03 ms | 1.92× | 63.08 ms | 1.31× |
| 512 tokens | 695.70 ms | 68.19 ms | 10.20× | 95.23 ms | 7.31× |
| 1,024 tokens | 1,642.10 ms | 103.57 ms | 15.85× | 131.86 ms | 12.45× |

Warm and restored produced the same tokens as native on all nine cases. The largest next-token score error was `5.44e-5`. Native, warm, and restored found 8 of 9 codes. All three answered `UNKNOWN` instead of `EQBU` on the 128-token late case. The no-notes control found 0 of 9.

Cache files were 3 MiB, 12 MiB, and 24 MiB. Process peak RSS was 2.98 GiB.
