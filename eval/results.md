| model | decision acc | extraction acc | runs with citations | p50 latency | tokens | cost (8 cases) |
|---|---|---|---|---|---|---|
| ollama:llama3.1-q4 | 62% | 38% | 62% | 16.5s | 19,633 | $0.0000 |

### ollama:llama3.1-q4
| case | expected | got | extraction | latency | errors |
|---|---|---|---|---|---|
| g01 | auto_approved | auto_approved | ok | 33.4s |  |
| g02 | declined | human_review | miss | 14.6s | model_error: Exceeded maximum output retries (2) |
| g03 | needs_info | needs_info | ok | 18.3s | llm_disagreement: model said decline, engine decided needs_info |
| g04 | human_review | declined | miss | 27.3s |  |
| g05 | declined | declined | miss | 12.9s |  |
| g06 | human_review | human_review | ok | 12.8s |  |
| g07 | auto_approved | auto_approved | miss | 11.9s | llm_disagreement: model said review, engine decided auto_approved |
| g08 | needs_info | human_review | miss | 25.3s | model_error: Exceeded maximum output retries (2) |
