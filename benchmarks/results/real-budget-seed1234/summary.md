# Real model token budget experiment summary

Run id: real-budget-seed1234

Fixed mixed workload, continuous strategy, concurrency 4, 20 requests
per run, 3 repeats, under docs/benchmark_protocol.md on a shared CPU
only machine. The accepted request set is identical at every budget
(the longest prompt, 24 tokens, fits the smallest budget of 32), so
differences are budget effects. Exploratory at this sample size.

| Budget | Steps mean | Wall s mean [min, max] | Output tok/s mean | Mean TTFT mean | Mean latency mean | Mean TPOT mean |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 42.0 | 13.02 [11.90, 14.70] | 12.39 | 6.1510 | 8.1303 | 0.2828 |
| 64 | 40.0 | 11.44 [10.81, 12.49] | 14.05 | 5.2530 | 6.9560 | 0.2433 |
| 8192 | 40.0 | 12.50 [11.82, 13.47] | 12.84 | 5.4543 | 7.3240 | 0.2671 |

Timeline request: main-mixed-01 (first long prompt request)

| Budget | Queue time per repeat | TTFT per repeat | Latency per repeat |
| --- | --- | --- | --- |
| 32 | 1.4155, 0.1942, 0.1774 | 2.1735, 0.5763, 0.3955 | 3.6940, 2.6808, 2.3688 |
| 64 | 0.1881, 0.1197, 0.1904 | 0.4068, 0.4014, 0.5051 | 2.4799, 2.2041, 2.4100 |
| 8192 | 0.1240, 0.1826, 0.1925 | 0.4909, 0.4010, 0.5859 | 2.4143, 2.0873, 2.2106 |

All checks passed: True
