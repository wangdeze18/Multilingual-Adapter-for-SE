# Results of Multilingual adapter tuning for CodeBERT and GraphCodeBERT

| Model    |     Ruby | JavaScript | Java | Go | PHP | Python | Overall (MRR) |
| :-: |  :-: |  :-: |  :-: |  :-: |  :-: |  :-: |  :-: |
| CodeBERT   |      67.7   |     61.6 |  67.6 |  88.5 |  62.9 | 67.6 | 69.3|
| *m*CodeBERT|      73.2   |     64.3 |  69.7 |  88.5 |  63.5 | 67.8 | 71.2|
| *m*Adapter-CodeBERT(ours)     |  71.3      |  63.7    | 68.5  |  87.9   |  62.2   | 66.7   | 70.1  |
| GraphCodeBERT   |     70.8    |   64.4   | 69.3  |  **89.4**   |   64.8  |  69.2  |  71.3 |
| *m*GraphCodeBERT|      73.8   |   66.0   | 71.0  |  89.4   |   64.6  |  69.5  |  72.4 |
| *m*Adapter-GraphCodeBERT(ours)     |   **75.0**      |  **67.0**    | **71.0**  |  89.1   |   **65.3**  |  **70.4**  |  **73.0** |
 
We denote multilingual fine-tuned models with the prefix *m*, as *m*CodeBERT is a multilingual model fine-tuned based on CodeBERT. *m*Adapter refers to models tuned with our multilingual adapter.


Please refer to [codebert](https://anonymous.4open.science/r/Multilingual-Adapter-for-SE-D360/code%20search/codebert/README.md) and [graphcodebert](https://anonymous.4open.science/r/Multilingual-Adapter-for-SE-D360/code%20search/graphcodebert/README.md) for implementation details.
