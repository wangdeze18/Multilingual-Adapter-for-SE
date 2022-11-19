# Results of Multilingual adapter tuning for CodeBERT and GraphCodeBERT

| Model    |     Ruby | JavaScript | Java | Go | PHP | Python | Overall (MRR) |
| :-: |  :-: |  :-: |  :-: |  :-: |  :-: |  :-: |  :-: |
| CodeBERT   |      67.7   |     61.6 |  67.6 |  88.5 |  62.9 | 67.6 | 69.3|
| *m*CodeBERT|      73.2   |     64.3 |  69.7 |  88.5 |  63.5 | 67.8 | 71.2|
| *m*Adapter-CodeBERT(ours)     |        |      |   |     |     |    |   |
| GraphCodeBERT   |     70.8    |   64.4   | 69.3  |  89.4   |   64.8  |  69.2  |  71.3 |
| *m*GraphCodeBERT|      73.8   |   66.0   | 71.0  |  89.4   |   64.6  |  69.5  |  72.4 |
| *m*Adapter-GraphCodeBERT(ours)     |         |      |   |     |     |    |   |
 
We denote multilingual fine-tuned models with the prefix *m*, as *m*CodeBERT is a multilingual model fine-tuned based on CodeBERT. *m*Adapter refers to models tuned with our multilingual adapter.
