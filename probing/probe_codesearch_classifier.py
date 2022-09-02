# train and evaluate classifier for each probing task
import sys, csv
import argparse
import json

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn import metrics
from senteval_tool import MLP


def load_data():
    with open(labels_file, 'r', encoding='utf-8', errors='ignore') as labl_reader:
        with open(feats_file, 'r') as feat_reader:

            feat_dim = -1
            cat2id = {}
            id2cat = {}

            train_X = []
            train_y = []
            valid_X = []
            valid_y = []
            test_X = []
            test_y = []

            while True:
                label_line = labl_reader.readline()
                if not label_line:
                    break

                feat_jsonl = feat_reader.readline()
                feat_jsonl = json.loads(feat_jsonl)

                # # ********************************************************
                # # **************** FEATURE TOKENS CHECK ******************
                # # ********************************************************
                # print("Length of tokens", len(feat_jsonl['features']))
                # for ix_ in range(len(feat_jsonl['features'])):
                #     print((feat_jsonl['features'][ix_]['token']), end=' ')
                # # ********************************************************

                # *****************
                # average the layer['values'] of every token
                # token_index 0 => layer['values'] => torch.Size([768])
                # token_index 1 => layer['values'] => torch.Size([768])
                # add each item in torch.Size([768]) across all token indices and divide by number of token indices
                # *****************

                all_X = []
                for token_index in range(len(feat_jsonl['features'])):
                    for layer in feat_jsonl['features'][token_index]['layers']:
                        if layer['index'] == eval_layer:
                            all_X.append(layer['values'])

                X = [float(sum(col)) / len(col) for col in zip(*all_X)]

                # *****************

                assert (X is not None)
                if feat_dim < 0: feat_dim = len(X)

                split, label, text = label_line.split('\t', 2)
                if label not in cat2id:
                    cat2id[label] = len(id2cat)
                    id2cat[cat2id[label]] = label
                y = cat2id[label]

                if split == 'tr':
                    train_X.append(X)
                    train_y.append(y)
                elif split == 'va':
                    valid_X.append(X)
                    valid_y.append(y)
                elif split == 'te':
                    test_X.append(X)
                    test_y.append(y)

    train_X = np.array(train_X, dtype=np.float32)
    valid_X = np.array(valid_X, dtype=np.float32)
    test_X = np.array(test_X, dtype=np.float32)

    train_y = np.array(train_y)
    valid_y = np.array(valid_y)
    test_y = np.array(test_y)

    # print('loaded %d/%d/%d samples; %d labels;'%(train_X.shape[0], valid_X.shape[0], test_X.shape[0], len(cat2id)))
    return train_X, train_y, valid_X, valid_y, test_X, test_y, feat_dim, len(cat2id), cat2id, id2cat


def classify_and_predict(train_X, train_y, dev_X, dev_y, test_X, test_y, feat_dim, num_classes):
    classifier_config = {'nhid': nhid, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 50,
                         'dropout': dropout}
    regs = [10 ** t for t in range(-5, -1)]
    props, scores = [], []

    # hyper-parameter optimization
    for reg in regs:
        clf = MLP(classifier_config, inputdim=feat_dim, nclasses=num_classes, l2reg=reg, seed=seed, cudaEfficient=True)
        clf.fit(train_X, train_y, validation_data=(dev_X, dev_y))
        scores.append(round(100 * clf.score(dev_X, dev_y), 2))
        props.append([reg])
    opt_prop = props[np.argmax(scores)]
    dev_acc = np.max(scores)

    # training
    classifier_config = {'nhid': nhid, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 50,
                         'dropout': dropout}
    clf = MLP(classifier_config, inputdim=feat_dim, nclasses=num_classes, l2reg=opt_prop[0], seed=seed,
              cudaEfficient=True)
    clf.fit(train_X, train_y, validation_data=(dev_X, dev_y))

    # testing
    test_acc = round(100 * clf.score(test_X, test_y), 2)

    # to get predictions use id2cat[]
    predictions = clf.predict(test_X)

    # writing orig and pred values to csv
    orig = [int(item) for item in test_y.tolist()]
    pred = [int(item[0]) for item in predictions.tolist()]

    orig = [id2cat[item] for item in orig]
    pred = [id2cat[item] for item in pred]

    orig_pred = (zip(orig, pred))
    # orig = [int(item[0]) if len(item) > 1 else int(item) for item in orig]
    # pred = [int(item[0]) if len(item) > 1 else int(item) for item in pred]
    # test_acc = metrics.r2_score(orig, pred)

    outpatx = sys.path[0] + '/outputs/' + task_code + '/' + 'code_search' + '/' + model_kind + '+' + head + '_' + str(
        label_count) + '_' + shuffle_kind + '_' + str(eval_layer) + '.csv'
    outpath = Path(outpatx)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    with open(outpath, 'w+') as wf:
        csv_writer = csv.writer(wf)
        csv_writer.writerow(['orig', 'pred'])

        for orig_item, pred_item in orig_pred:
            csv_writer.writerow([orig_item, pred_item])

    # confusion matrix
    plt.figure()
    labels = list(id2cat.values())
    labels.sort()

    cf_matrix = metrics.confusion_matrix(orig, pred, labels=labels)
    ax = sns.heatmap(cf_matrix, cmap='RdYlGn', annot=True, yticklabels=labels, fmt='g', square=2, linecolor="white")

    ax.figure.subplots_adjust(left=0.3)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('PRED Labels')
    ax.set_ylabel('ORIG Labels')

    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    ax.figure.savefig(outpatx[:-4] + '_CFMX.png')
    plt.close()

    print(test_acc, end='\t')
    return test_acc


if __name__ == "__main__":

    shuffle_kinds = ['ORIG']
    model_kinds = {
        "unixcoder_multilin": 13,
        "unixcoder_multilin_adapter": 13,
    }
    # "BERT_finetune_adapter":    13,
    # "BERT_finetune_init_adapter":   13,
    # "BERT_augmented": 13,
    # "BERT_adapter_augmented": 13,
    # "CodeBERT_augmented": 13,
    # "CodeBERT_adapter_augmented": 13,
    # "GraphCodeBERT_augmented": 13,

    # "unixcoder_java": 13,
    # "unixcoder_ruby": 13,
    # "unixcoder_php": 13,
    # "unixcoder_python": 13,
    # "unixcoder_go": 13,
    # "unixcoder_javascript": 13,
    # "unixcoder_adapter_java": 13,
    # "unixcoder_adapter_ruby": 13,
    # "unixcoder_adapter_php": 13,
    # "unixcoder_adapter_python": 13,
    # "unixcoder_adapter_go": 13,
    # "unixcoder_adapter_javascript": 13,

    label_counts = ['10k']
    task_codes = [
        'AST']  # ['AST', 'CPX', 'CSC', 'JBL', 'JFT', 'JMB', 'LEN', 'MXN', 'NML', 'NMS', 'NPT', 'OCT', 'OCU', 'REA', 'SCK', 'SRI', 'SRK', 'TAN', 'TYP', 'VCT', 'VCU']
    nhids = [0,128,256]  # number of hidden layers

    for task_code in task_codes:

        print(f'\n****\n{task_code}\n****\n')
        for label_count in label_counts:
            for nhid in nhids:
                head = ('MLP' if nhid != 0 else 'LIN')

                for model_kind in list(model_kinds.keys()):
                    eval_layers_count = model_kinds.get(model_kind, 1)
                    spacing = (' ' * (20 - len((model_kind + "+" + head)))) + '\t'

                    for shuffle_kind in shuffle_kinds:
                        print('\n' + model_kind + "+" + head + spacing + label_count + '\t' + shuffle_kind, end='\t')

                        for eval_layer in range(eval_layers_count):
                            labels_file = (sys.path[
                                               0] + '/data/datasets_' + task_code + '/' + task_code + '_' + shuffle_kind + '_' + label_count + '.txt')
                            feats_file = (sys.path[
                                              0] + '/data/datasets_' + task_code + '/' + shuffle_kind + '/' + 'code_search' + '/' + model_kind + '_features_' + label_count + '.json')
                            dropout = 0.1
                            seed = 42

                            train_X, train_y, dev_X, dev_y, test_X, test_y, feat_dim, num_classes, cat2id, id2cat = load_data()
                            test_acc = classify_and_predict(train_X, train_y, dev_X, dev_y, test_X, test_y, feat_dim,
                                                            num_classes)
            ##
            print('\n\n')

