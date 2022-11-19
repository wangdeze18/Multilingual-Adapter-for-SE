# Code Summarization
 
 multilingual adapter tuning for CodeBERT and GraphCodeBERT


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
 
