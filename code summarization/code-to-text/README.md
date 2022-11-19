# Results

| Model    |     Ruby | JavaScript | Java | Go | PHP | Python | Overall (BLEU-4) |
| :-: |  :-: |  :-: |  :-: |  :-: |  :-: |  :-: |  :-: |
| CodeBERT   |      12.53   |     13.86 |  18.72 |  18.15 |  25.48 | 18.25 | 17.83|
| *m*CodeBERT|      14.75   |     **15.80** |  **20.11** |  18.77 |  **26.23** | 18.71 | 19.06|
| *m*Adapter-CodeBERT(ours)     |      14.95   |     15.42 |  19.34 | 18.71  |  25.61 | 19.76 | 18.97|
| GraphCodeBERT   |      12.62   |     14.79 |  19.22 |  18.40 |  25.45 | 18.02 | 18.08|
| *m*GraphCodeBERT|      14.95   |     15.79 |  19.91 |  18.92 |  26.15 | 18.90 | 19.10|
| *m*Adapter-GraphCodeBERT(ours)     |      **15.09**  |15.40 |     19.55 |  **19.05** |  26.05 |  **19.98** | **19.19**|
 
We denote multilingual fine-tuned models with the prefix *m*, as *m*CodeBERT is a multilingual model fine-tuned based on CodeBERT. *m*Adapter refers to models tuned with our multilingual adapter.

# Multilingual adapter tuning for CodeBERT and GraphCodeBERT
1. Download the dataset

```
wget https://github.com/microsoft/CodeXGLUE/raw/main/Code-Text/code-to-text/dataset.zip
unzip dataset.zip
rm dataset.zip
cd dataset
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/ruby.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/javascript.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/go.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/php.zip

unzip python.zip
unzip java.zip
unzip ruby.zip
unzip javascript.zip
unzip go.zip
unzip php.zip
rm *.zip
rm *.pkl

python preprocess.py
rm -r */final
cd ..
```


2. Fine-tuning the multlingual adapter and evaluation 

  
```
sh ./run_multilingual_adapter.sh
```

GraphCodeBERT: Follow the exact instructions followed for CodeBERT, just replace the "microsoft/codebert-base" with "microsoft/graphcodebert-base". 
 
