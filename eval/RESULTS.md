# Retrieval results (test split, retriever=hybrid)

Questions: 37 (29 in-scope, 8 out-of-scope). Thresholds were tuned on the dev split only.

- hit@1 79.3%, hit@3 93.1% (95% CI 83-100), MRR 0.85, nDCG@3 0.78
- false refusals 0.0% of in-scope; correct refusals 87.5% of out-of-scope

| type | n | hit@1 | hit@3 | MRR | nDCG@3 |
|---|---|---|---|---|---|
| known | 14 | 100% | 100% | 1.00 | 0.92 |
| inferred | 11 | 45% | 82% | 0.61 | 0.57 |
| overview | 4 | 100% | 100% | 1.00 | 0.85 |
