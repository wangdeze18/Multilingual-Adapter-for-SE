# Results

| Model    |     Ruby | JavaScript | Java | Go | PHP | Python | Overall (BLEU-4) |
| :-: |  :-: |  :-: |  :-: |  :-: |  :-: |  :-: |  :-: |
| CodeBERT   |      12.53   |     13.86 |  18.72 |  18.15 |  25.48 | 18.25 | 17.83|
| *m*CodeBERT|      14.75   |     **15.80** |  **20.11** |  **18.77** |  **26.23** | 18.71 | **19.06**|
| *m*Adapter-CodeBERT(ours)     |      **14.95**   |     15.42 |  19.34 | 18.71  |  25.61 | **19.76** | 18.97|
| GraphCodeBERT   |      0.9295   |     0.9122 |  0.8912 |  0.8763 |  0.8599 | 0.8494 | 0|
| *m*GraphCodeBERT|      0.9094   |     0.8888 |  0.8638 |  0.8461 |  0.8267 | 0.8142 | 0|
| *m*Adapter-GraphCodeBERT(ours)     |      62.4   |     49.1 |  52.6 |   |  44.6 | 50.5 | 0|
 


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
 You can download the [pre-trained model](https://drive.google.com/file/d/1jKwfWiCO6izkcOtHabmUWRsI_YGSo_C2/view?usp=sharing) directly for check.
 
