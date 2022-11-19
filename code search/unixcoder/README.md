# Low-resource experiment on code search

We randomly sample the datasets in each programming languages, and choose 100, 200, 500 and 1000 samples for each language. Then, we insert the adapter into UniXcoder and evaluate the model on the combinations of these data. We vary the random seed, repeat the experiment several times, and average the results.

| Training samples    |     Ruby | JavaScript | Java | Go | PHP | Python | Overall |
| :-: |  :-: |  :-: |  :-: |  :-: |  :-: |  :-: |  :-: |
| 6*100   |      0.9295   |     0.9122 |  0.8912 |  0.8763 |  0.8599 | 0.8494 | 0|
| 6*200     |      0.9094   |     0.8888 |  0.8638 |  0.8461 |  0.8267 | 0.8142 | 0|
| 6*500     |      62.4   |     49.1 |  52.6 |   |  44.6 | 50.5 | 0|
| 6*1,000     |      64.8   |     52.1 |  55.8 |  72.2 | 47.7  | 54.0 | 57.8 |
| 908,224     |      77.3  |     70.2 |  73.5 |  90.9 |  67.1 | 72.7 | 75.3 |




# Code Search

1. Download the dataset


```
mkdir dataset && cd dataset
wget https://github.com/microsoft/CodeBERT/raw/master/GraphCodeBERT/codesearch/dataset.zip
unzip dataset.zip && rm -r dataset.zip && mv dataset CSN && cd CSN
bash run.sh 
cd ../..
```


2. Fine-tuning the multlingual adapter

   You can also skip this step by downloading the [pre-trained model](https://drive.google.com/file/d/10fFosyNVEJiAwdsfPiwhWI5o14fmU69Q/view?usp=sharing) directly.
```
python  run_multilin_adapter.py     --output_dir saved_models     --model_name_or_path microsoft/unixcoder-base      --do_train     --train_data_dir dataset/CSN  --test_data_dir dataset/CSN    --eval_data_dir dataset/CSN     --codebase_file dataset/CSN     --num_train_epochs 10     --code_length 256     --nl_length 128     --train_batch_size 64     --eval_batch_size 64     --learning_rate 5e-5     --seed 123456  2>&1| tee saved_models/multilin_adapter_train.log
```

3. Evaluation
```
python   run_multilin_adapter.py     --output_dir saved_models     --model_name_or_path microsoft/unixcoder-base      --do_test     --train_data_dir dataset/CSN     --test_data_dir dataset/CSN     --codebase_file dataset/CSN     --num_train_epochs 10     --code_length 256     --nl_length 128     --train_batch_size 64     --eval_batch_size 64     --learning_rate 1e-4     --seed 123456  2>&1| tee saved_models/multilin_adapter_test.log
```

4. Other baselines for check
```
./run_adapter.py  # Fine-tuning the monolingual model with adapter
```


