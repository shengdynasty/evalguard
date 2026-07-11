# Real data cache

These files are REAL public data, fetched once via the Hugging Face
`datasets-server` REST API (https://datasets-server.huggingface.co) and cached
here so the real-data demo is fully reproducible offline.

- `openbookqa.jsonl` — 300 rows of the OpenBookQA MCQ science benchmark
  (allenai/openbookqa, config main, split train, rows 0-299). Each line:
  {"id","question","answer"} where answer is the text of the correct choice.
- `ag_news.jsonl` — 200 rows of the AG News text corpus
  (fancyzhx/ag_news, split train, rows 0-199). Each line: {"id","text"}.

Provenance note: direct datasets.load_dataset(...) is blocked in the build
sandbox (the outbound proxy returns 403 for huggingface.co hosts, so the
library's HTTP/xet backend cannot reach the CDN). The rows were instead pulled
through the datasets-server /rows JSON endpoint, which was reachable, and
normalized to the JSONL above. This is genuinely the public OpenBookQA / AG News
data, not synthetic.
