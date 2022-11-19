# Multilingual-Adapter-for-SE

This repository contains code for paper "One Adapter for All Programming Languages? Adapter Tuning for Multilingual Tasks in Software Engineering". 


# How to use
We apply adapter tuning to code pre-trained models UniXcoder and CodeT5 for fine-tuning. 
  
  Please refer to the subdirectory of tasks for more details ([code summarization](https://github.com/wangdeze18/Multilingual-Adapter-for-SE/tree/main/code%20summarization) and [code search](https://github.com/wangdeze18/Multilingual-Adapter-for-SE/tree/main/code%20search/unixcoder)) !

 

# Datasets and Models 

The data statistics of original CodeSearchNet is shown in the table below. The dataset contains about 2 million pairs of function-documentation pairs and about another 4 million functions without an associated documentation.
| Programming Language | w/ documentation |  All  |
| :------------------- | :------: | :----: |
| Python               | 503,502  | 1,156,085 | 
| PHP                  | 717,313  | 977,821 | 
| Go                   | 347,789  | 726,768  | 
| Java                 | 542,991  | 1,569,889  | 
| JavaScript           | 157,988  | 1,857,835  | 
| Ruby                 |  57,393  | 164,048  | 



The dataset we use are futher filtered by [CodeXGLUE](https://github.com/microsoft/CodeXGLUE).
| Programming Language | Training |  Dev   |  Test  |
| :------------------- | :------: | :----: | :----: |
| Python               | 251,820  | 13,914 | 14,918 |
| PHP                  | 241,241  | 12,982 | 14,014 |
| Go                   | 167,288  | 7,325  | 8,122  |
| Java                 | 164,923  | 5,183  | 10,955 |
| JavaScript           |  58,025  | 3,885  | 3,291  |
| Ruby                 |  24,927  | 1,400  | 1,261  |


**We also provide pre-trained models fine-tuned by our approach to verify the results.**

# Acknowledgement
Our implementation is adapted from [CodeXGLUE](https://github.com/microsoft/CodeXGLUE), [UniXcoder](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder), and [CodeT5](https://github.com/salesforce/CodeT5) for the implementation of pre-trained models.
