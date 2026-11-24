# Real model main experiment summary

Run id: real-main-seed1234

Closed batch runs under docs/benchmark_protocol.md on a shared CPU
only machine. Sample: 20 requests per run, 3 repeats, 60 finished
requests per continuous point, below the plan target of about 100,
so every figure is exploratory. Serial runs once per workload and
repeat (it has no concurrency variable) and pairs with each
continuous concurrency of the same repeat. Means are across repeats,
with the repeat range in brackets.

| Workload | Strategy | Concurrency | Wall s mean [min, max] | Output tok/s mean | Mean TTFT mean | Mean latency mean | Mean TPOT mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| short | serial | 1 | 27.39 [25.83, 28.40] | 5.85 | 13.2452 | 14.3988 | 0.1648 |
| short | continuous | 1 | 25.70 [25.01, 26.41] | 6.23 | 12.4427 | 13.5387 | 0.1566 |
| short | continuous | 2 | 15.69 [14.99, 16.20] | 10.21 | 7.2517 | 8.5152 | 0.1805 |
| short | continuous | 4 | 10.56 [9.80, 11.00] | 15.19 | 4.7351 | 6.3491 | 0.2306 |
| short | continuous | 8 | 9.30 [7.78, 10.49] | 17.48 | 3.5028 | 5.8303 | 0.3325 |
| mixed | serial | 1 | 27.46 [26.80, 27.88] | 5.83 | 13.3742 | 14.5063 | 0.1617 |
| mixed | continuous | 1 | 28.24 [27.11, 29.09] | 5.67 | 13.4573 | 14.6121 | 0.1650 |
| mixed | continuous | 2 | 18.23 [17.71, 18.90] | 8.78 | 8.7397 | 10.1726 | 0.2047 |
| mixed | continuous | 4 | 11.49 [11.20, 12.03] | 13.95 | 5.0521 | 6.7148 | 0.2375 |
| mixed | continuous | 8 | 10.30 [9.70, 10.69] | 15.56 | 4.0870 | 6.7141 | 0.3753 |

Token comparisons all match: True
All checks passed: True
