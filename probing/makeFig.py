# coding:utf-8


import numpy as np
import matplotlib.pyplot as plt
import  os
#ad file
def ctoint(strlist):
    intlist =[]
    for i in range(len(strlist)):
        intlist.append(float(strlist[i]))
    return  intlist
def plot_figs(dict_fig1, dict_fig2,task):
    fig, ax = plt.subplots()
    x = np.arange(0, 11)
    for key, item in dict_fig1.items():
        #if  key[0:4] =='BERT':
        if  key[0:8] =='CodeBERT':
        #if key[0:13] == 'GraphCodeBERT':
        # if 'finetune_adapter+LIN' in key or 'finetuned_adapter+LIN' in key:
        #    ax.plot(x, item, label=key)
        # if  'finetune+LIN' in key or 'finetuned+LIN' in key:
        #    ax.plot(x, item, label=key)
        # if 'finetune' not in key and 'adapter' not in key:
            ax.plot(x, item, label=key)
    for key, item in dict_fig2.items():
        #if  key[0:4] =='BERT':
        if  key[0:8] =='CodeBERT':
        #if key[0:13] == 'GraphCodeBERT':
        # if 'finetune_adapter+LIN' in key or 'finetuned_adapter+LIN' in key:
        #    ax.plot(x, item, label=key)
        # if  'finetune+LIN' in key or 'finetuned+LIN' in key:
        #    ax.plot(x, item, label=key)
        # if 'finetune' not in key and 'adapter' not in key:
            ax.plot(x, item, label=key)
    plt.title(task)
    plt.xlabel('Layer')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()
def plot_fig(dict_fig,task):
    fig,ax = plt.subplots()
    x = np.arange(0, 11)
    for key,item in dict_fig.items():

        #if  key[0:4] =='BERT':
        #if  key[0:8] =='CodeBERT':
        #if key[0:13] == 'GraphCodeBERT':
        #if 'finetune_adapter+LIN' in key or 'finetuned_adapter+LIN' in key:
        #    ax.plot(x, item, label=key)
        #if  'finetune+LIN' in key or 'finetuned+LIN' in key:
        #    ax.plot(x, item, label=key)
        #if 'finetune' not in key and 'adapter' not in key:
            ax.plot(x, item, label=key)
    plt.title(task)
    plt.xlabel('Layer')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()

task_codes  = ['AST']#'TYP', 'CPX', 'LEN','TYP'
augmented_dir = 'augmented'

def getcontent(file_name):
    dict1 ={}
    dict2={}
    dict3 = {}
    if os.path.exists(file_name):
        print(file_name)
        with open(file_name) as f:
            lines = f.readlines()
            #print(lines)
            for line in lines:
                data = line.strip().split()
                if len(data) > 10:
                    row_name = data[0]
                    int_list = ctoint(data[4:])
                    if data[1] == '100':
                        dict1[row_name] = int_list
                    elif data[1] == '1k':
                        dict2[row_name] = int_list
                    else:
                        dict3[row_name] = int_list
    return dict1, dict2, dict3
for task in task_codes:
    dict_10k = {}
    dict_100 = {}
    dict_1000 = {}
    aug_dict_10k = {}
    aug_dict_100 = {}
    aug_dict_1000 = {}
    file_name = task
    dict_100,dict_1000,dict_10k = getcontent(file_name)

    if  os.path.exists(augmented_dir):
        augmented_file = os.path.join(augmented_dir,file_name)
        aug_dict_100,aug_dict_1000,aug_dict_10k = getcontent(augmented_file)

    #plot_fig(dict_10k)
    plot_fig(aug_dict_10k,task)
    plot_figs(dict_10k,aug_dict_10k,task)
