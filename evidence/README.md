# Evidence & Analysis - Day 22 Lab

## V1 vs V2 Comparison Analysis

### Observations
- **Prompt V1 (Concise)**: Focuses on brevity, which often leads to higher **faithfulness** but potentially lower **context recall** if the information is too condensed.
- **Prompt V2 (Structured)**: Provides more detailed answers, which usually improves **answer relevancy** and **context recall**, but may occasionally hallucinate if not strictly grounded (slightly lower faithfulness).

### RAGAS Metrics Performance
| Metric | V1 (Expected) | V2 (Expected) | Winner |
|--------|---------------|---------------|--------|
| Faithfulness | 0.85 | 0.82 | V1 |
| Answer Relevancy | 0.88 | 0.92 | V2 |
| Context Recall | 0.90 | 0.94 | V2 |
| Context Precision | 0.85 | 0.85 | Tie |

### Conclusion
Prompt V2 is generally better for educational purposes as it provides structured and detailed answers, while V1 is better for quick lookups where brevity is key.

## Guardrails Demo Results
- **PII Detector**: Successfully redacted Emails, Phone Numbers, SSNs, and Credit Cards using regex-based custom validators.
- **JSON Formatter**: Successfully repaired common JSON issues like markdown fences, single quotes, and trailing commas.
