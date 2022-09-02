# coding:utf-8

"""
Author: roguesir
Date: 2017/8/30
GitHub: https://roguesir.github.com
Blog: http://blog.csdn.net/roguesir
"""

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
    x = np.arange(0, 12)
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
    x = np.arange(0, 12)
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
'''
x1 = [20, 33, 51, 79, 101, 121, 132, 145, 162, 182, 203, 219, 232, 243, 256, 270, 287, 310, 325]
y1 = [49, 48, 48, 48, 48, 87, 106, 123, 155, 191, 233, 261, 278, 284, 297, 307, 341, 319, 341]
x2 = [31, 52, 73, 92, 101, 112, 126, 140, 153, 175, 186, 196, 215, 230, 240, 270, 288, 300]
y2 = [48, 48, 48, 48, 49, 89, 162, 237, 302, 378, 443, 472, 522, 597, 628, 661, 690, 702]
x3 = [30, 50, 70, 90, 105, 114, 128, 137, 147, 159, 170, 180, 190, 200, 210, 230, 243, 259, 284, 297, 311]
y3 = [48, 48, 48, 48, 66, 173, 351, 472, 586, 712, 804, 899, 994, 1094, 1198, 1360, 1458, 1578, 1734, 1797, 1892]
x = np.arange(20, 350)
l1 = plt.plot(x1, y1, 'r--', label='type1')
l2 = plt.plot(x2, y2, 'g--', label='type2')
l3 = plt.plot(x3, y3, 'b--', label='type3')
plt.plot(x1, y1, 'ro-', x2, y2, 'g+-', x3, y3, 'b^-')
plt.title('The Lasers in Three Conditions')
plt.xlabel('row')
plt.ylabel('column')
plt.legend()
plt.show()

'''