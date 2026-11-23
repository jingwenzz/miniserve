# Real model pilot summary

Run id: real-pilot-seed1234

Wall clock observations on a shared CPU only machine under the
protocol in docs/benchmark_protocol.md. The pilot checks the
measurement path; it is not a headline performance result.

| Strategy | Finished | Failed | Output tokens | Steps | Wall seconds | Mean TTFT | Mean latency | Mean TPOT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| serial | 20 | 0 | 160 | 160 | 27.4936 | 13.0717 | 14.2577 | 0.1694 |
| continuous | 20 | 0 | 160 | 40 | 13.9019 | 6.1622 | 8.2292 | 0.2953 |

Workload check passed: True
Manual timing check passed: True
