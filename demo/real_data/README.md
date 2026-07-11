# Real data cache

These files are REAL public data, fetched once via the Hugging Face
`datasets-server` REST API (https://datasets-server.huggingface.co) and cached
here so the real-data demo is fully reproducible offline.

- `openbookqa.jsonl` — 300 rows of the OpenBookQA MCQ science benchmark
  (allenai/openbookqa, config main, split train, rows 0-299). Each line:
  {"id","question","answer"} where answer is the text of the correct choice.
- `ag_news.jsonl` — 200 rows of the AG News text corpus
  (fancyzhx/ag_news, split train, rows 0-199). Each line: {"id","text"}.

- `sciq.jsonl` — 300 rows of the SciQ science benchmark
  (allenai/sciq, split train, rows 0-299). Each line:
  {"id","question","answer"} with distinctive science-term answers.
- `gsm8k.jsonl` — 300 rows of the GSM8K math benchmark
  (openai/gsm8k, config main, split train, rows 0-299). Each line:
  {"id","question","answer"} where answer is the final numeric result.

Provenance note: data was fetched via the HuggingFace datasets-server
REST API (/rows endpoint) and normalized to the JSONL format above.
This is genuinely the public data from the respective HuggingFace datasets,
not synthetic.
