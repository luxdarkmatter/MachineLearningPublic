'''
    Last edited (1/8/20)
'''
#---------------------------------------------------------------------------------------------------------------
#	Required packages
import os
# set the backend to tensorflow
os.environ['KERAS_BACKEND']='tensorflow'
import numpy as np
from math import log
import csv
# tensor functionality uses K
from tensorflow.keras import backend as K
from tensorflow.keras import initializers as ki
import tensorflow as tf
import time
from sklearn.utils import shuffle as shuffle2

import bisect
from random import shuffle
import h5py
from typing import List

from lux_ml.mlp import MLP
from lux_ml.utils import generate_binary_training_testing_data
from lux_ml.utils import print_output, get_system_info, make_folder
import lux_ml.mi as mi


#---------------------------------------------------------------------------------------------------------------
#   preprocess data
#   steps include:
#       1)  pull in signal and background data
#       2)  normalize the dataset so that (x - mu)/sigma, mu = 0, sigma = 1
#       3)  rebalance weights so that sum(signal) = sum(background)
#       4)  split into training/testing sets
#           4a) check that splitting creates subsets which are statistically indistinguishable
#       5)  split testing set into subsets and save results to h5py files.
#---------------------------------------------------------------------------------------------------------------
def preprocess_data(
    data_folder: str,
    signal_file: str,
    background_file: str,
    output_folder: str,
    var_set: List[int],
    weight: int=-1,
    test_split: float=.2,
    normalized: bool=False,
    balanced: bool=False,
    num_subsets: int=1
):
    preprocessing_start = time.time()
    #   first we generate the folder
    make_folder(output_folder)
    # keep track of what output_info we are printing
    print_index, print_folder = 0, output_folder+"preprocesing_output.csv"
    # we want to store output information into a txt file
    output_info = [["---------------------------------------------------------------------------------"],
                   ["                                 Preprocessing Data                              "],
                   ["---------------------------------------------------------------------------------"],
                   [" Parameters:                                   "],
                   ["   a) data_folder:      %s" % data_folder],
                   ["   b) signal_file:      %s" % signal_file],
                   ["   c) background_file:  %s" % background_file],
                   ["   d) output_folder:    %s" % output_folder],
                   ["   e) var_set:          %s" % var_set],
                   ["   f) weight:           %s" % weight],
                   ["   g) test_split:       %s" % test_split],
                   ["   h) normalized:       %s" % normalized],
                   ["   i) balanced:         %s" % balanced],
                   ["   j) num_subsets:      %s" % num_subsets],
                   ["---------------------------------------------------------------------------------"]]
    print_index = print_output(output_info,print_index,print_folder)
    start = time.time()
    output_info.append([" Step 1:  Loading data..."])
    data = generate_binary_training_testing_data(
        data_folder+signal_file,
        data_folder+background_file,
        labels=['arr_0','arr_0'],
        testing_ratio=0.0,
        symmetric_signals=False,
        var_set=var_set,
        weight=weight
    )
    N = len(data[0])
    ind_s = [i for i in range(N) if data[1][i][0] == 1]
    ind_b = [i for i in range(N) if data[1][i][0] != 1]
    N_s = len(ind_s)
    N_b = len(ind_b)
    output_info.append(["---------------------------------------------------------------------------------"])
    output_info.append(["   Number of signal events:        %s" % N_s])
    output_info.append(["   Number of background events:    %s" % N_b])
    output_info.append(["   Total number of events:         %s" % N])
    print_index = print_output(output_info,print_index,print_folder)
    # construct arrays for data, answer, and weights
    data_in = np.asarray(data[0])
    data_answer = np.asarray(data[1])
    # make a copy of the data for normalization
    if weight != -1:
        data_weights = np.asarray(data[2])
    end = time.time()
    output_info.append(["---------------------------------------------------------------------------------"])
    output_info.append([" Step 1 finished:  took %s." % (end-start)])
    output_info.append(["---------------------------------------------------------------------------------"])
    print_index = print_output(output_info,print_index,print_folder)
    # find normalization parameters
    ###########################################################################
    # Step 4a
    starta = time.time()
    output_info.append([" Step 2:  Finding norm. parameters..."])
    output_info.append(["---------------------------------------------------------------------------------"])
    print_index = print_output(output_info,print_index,print_folder)
    if not normalized:
        means = np.mean(data_in,axis=0)
        stds = np.std(data_in,axis=0)
        output_info.append(["    Norm params:  mean,    std dev"])
        for l in range(len(means)):
            output_info.append(["      var %s):    %s,  %s" % (l,means[l],stds[l])])
    else:
        output_info.append(["   Data already normalized."])
    enda = time.time()
    output_info.append(["---------------------------------------------------------------------------------"])
    output_info.append([" Step 2 finished:  took %s." % (enda-starta)])
    output_info.append(["---------------------------------------------------------------------------------"])
    print_index = print_output(output_info,print_index,print_folder)
    ###########################################################################
    # Step 4b
    startb = time.time()
    output_info.append([" Step 3:  Normalizing input data..."])
    output_info.append(["---------------------------------------------------------------------------------"])
    print_index = print_output(output_info,print_index,print_folder)
    if not normalized:
        data_in = (data_in - means)/stds
    endb = time.time()
    output_info.append([" Step 3 finished:  took %s." % (endb-startb)])
    output_info.append(["---------------------------------------------------------------------------------"])
    output_info.append([" Step 3a: Saving normalization parameters..."])
    startnorm = time.time()
    output_info.append(["---------------------------------------------------------------------------------"])
    norm_params = [["var %s",means[i],stds[i]] for i in range(len(means))]
    with open(output_folder+"norm_params.csv","w") as file:
        writer = csv.writer(file,delimiter=",")
        writer.writerows(norm_params)
    endnorm = time.time()
    output_info.append([" Step 3a finished: took %s." % (endnorm - startnorm)])
    output_info.append(["---------------------------------------------------------------------------------"])
    ###########################################################################
    # Step 2
    start = time.time()
    output_info.append([" Step 4:  Separating weights..."])
    output_info.append(["---------------------------------------------------------------------------------"])
    print_index = print_output(output_info,print_index,print_folder)
    start = time.time()
    if weight != -1:
        # weight statistics
        w_s = data_weights[ind_s]
        w_b = data_weights[ind_b]
        output_info.append(["   Sum of signal weights:     %s" % np.sum(w_s)])
        output_info.append(["   Sum of background weights: %s" % np.sum(w_b)])
    end = time.time()
    
    output_info.append(["---------------------------------------------------------------------------------"])
    output_info.append([" Step 4 finished:  took %s." % (end - start)])
    output_info.append(["---------------------------------------------------------------------------------"])
    print_index = print_output(output_info,print_index,print_folder)
    ###########################################################################
    # Step 3
    start = time.time()
    output_info.append([" Step 5:  Generating training/testing split..."])
    output_info.append(["---------------------------------------------------------------------------------"])
    output_info.append(["   Shuffling signal/background indices..."])
    shuffle(ind_s)
    shuffle(ind_b)
    signal_split_val = int(len(ind_s)*(1-test_split))
    background_split_val = int(len(ind_b)*(1-test_split))
    output_info.append(["   Signal split index:     %s" % signal_split_val])
    output_info.append(["   Background split index: %s" % background_split_val])
    train_ind_s = ind_s[:signal_split_val]
    train_ind_b = ind_b[:background_split_val]
    test_ind_s = ind_s[signal_split_val:]
    test_ind_b = ind_b[background_split_val:]
    output_info.append(["   Number of training signal events:       %s" % len(train_ind_s)])
    output_info.append(["   Number of testing  signal events:       %s" % len(test_ind_s)])
    output_info.append(["   Number of training background events:   %s" % len(train_ind_b)])
    output_info.append(["   Number of testing  background events:   %s" % len(test_ind_b)])
    print_index = print_output(output_info,print_index,print_folder)
    starta = time.time()
    output_info.append(["---------------------------------------------------------------------------------"])
    output_info.append(["   Step 5a:  Splitting training/testing data/weights..."])
    output_info.append(["---------------------------------------------------------------------------------"])
    train_s = data_in[train_ind_s]
    train_b = data_in[train_ind_b]
    test_s = data_in[test_ind_s]
    test_b = data_in[test_ind_b]
    if weight != -1:
        train_w_s = data_weights[train_ind_s]
        train_w_b = data_weights[train_ind_b]
        test_w_s = data_weights[test_ind_s]
        test_w_b = data_weights[test_ind_b]
    enda = time.time()
    output_info.append(["   Step 5a finished:  took %s." % (enda-starta)])
    print_index = print_output(output_info,print_index,print_folder)
    startb = time.time()
    output_info.append(["---------------------------------------------------------------------------------"])
    output_info.append(["   Step 5b:  Checking for statistical independence among split..."])
    output_info.append(["---------------------------------------------------------------------------------"])
    print_index = print_output(output_info,print_index,print_folder)
    train_s_means = np.mean(train_s,axis=0)
    train_b_means = np.mean(train_b,axis=0)
    test_s_means = np.mean(test_s,axis=0)
    test_b_means = np.mean(test_b,axis=0)
    train_s_stds = np.std(train_s,axis=0)
    train_b_stds = np.std(train_b,axis=0)
    test_s_stds = np.std(test_s,axis=0)
    test_b_stds = np.std(test_b,axis=0)
    output_info.append(["       Signal:     mean|train - test|,     std dev|train - test|"])
    for l in range(len(train_s_means)):
        output_info.append(["           var %s):    %.3e,  %.3e" % (l,np.abs(train_s_means[l]-test_s_means[l]),np.abs(train_s_stds[l]-test_s_stds[l]))])
    output_info.append(["       Background:     mean|train - test|,     std dev|train - test|"])
    for l in range(len(train_b_means)):
        output_info.append(["           var %s):    %.3e,  %.3e" % (l,np.abs(train_b_means[l]-test_b_means[l]),np.abs(train_b_stds[l]-test_b_stds[l]))])
    if weight != -1:
        sum_train_w_s = np.sum(train_w_s)
        sum_test_w_s = np.sum(test_w_s)
        sum_train_w_b = np.sum(train_w_b)
        sum_test_w_b = np.sum(test_w_b)
        output_info.append(["       Type:           sum training,   sum testing,   ratio"])
        output_info.append(["       (signal weights)         %s     %s      %s" % (sum_train_w_s,sum_test_w_s,float(sum_test_w_s)/sum_train_w_s)])
        output_info.append(["       (background weights)     %s     %s      %s" % (sum_train_w_b,sum_test_w_b,float(sum_test_w_b)/sum_train_w_b)])
    print_index = print_output(output_info,print_index,print_folder)
    output_info.append(["---------------------------------------------------------------------------------"])
    endb = time.time()
    output_info.append(["   Step 5b finished:  took %s." % (endb-startb)])
    output_info.append(["---------------------------------------------------------------------------------"])
    startc = time.time()
    output_info.append(["   Step 5c:  Balancing training weights..."])
    output_info.append(["---------------------------------------------------------------------------------"])
    if not balanced:
        if weight != -1:
            if sum_train_w_s < sum_train_w_b:
                train_w_s *= (sum_train_w_b/sum_train_w_s)
            else:
                train_w_b *= (sum_train_w_s/sum_train_w_b)    
            output_info.append(["   Sum of new training signal weights:     %s" % np.sum(train_w_s)])
            output_info.append(["   Sum of new training background weights: %s" % np.sum(train_w_b)])
    else:
        if weight != -1:
            output_info.append(["   Weights already balanced."])
    endc = time.time()
    output_info.append(["---------------------------------------------------------------------------------"])
    output_info.append(["   Step 5c finished:  took %s." % (endc-startc)])
    output_info.append(["---------------------------------------------------------------------------------"])
    print_index = print_output(output_info,print_index,print_folder)        
    startd = time.time()
    output_info.append(["   Step 5d:  Balancing testing weights..."])
    output_info.append(["---------------------------------------------------------------------------------"])
    if not balanced:
        if weight != -1:
            if sum_test_w_s < sum_test_w_b:
                test_w_s *= (sum_test_w_b/sum_test_w_s)
            else:
                test_w_b *= (sum_test_w_s/sum_test_w_b)    
            output_info.append(["   Sum of new testing signal weights:     %s" % np.sum(test_w_s)])
            output_info.append(["   Sum of new testing background weights: %s" % np.sum(test_w_b)])
    else:
        if weight != -1:
            output_info.append(["   Weights already balanced."])
    endd = time.time()
    output_info.append(["---------------------------------------------------------------------------------"])
    output_info.append(["   Step 5d finished:  took %s." % (endd-startd)])
    output_info.append(["---------------------------------------------------------------------------------"])
    print_index = print_output(output_info,print_index,print_folder) 
    starte = time.time()
    output_info.append(["   Step 5e:  Saving testing data..."])
    output_info.append(["---------------------------------------------------------------------------------"])       
    
    test_s_answer = np.asarray([1.0 for i in range(len(test_s))])
    test_b_answer = np.asarray([0.0 for i in range(len(test_b))])

    test_data = np.concatenate((test_s,test_b))
    test_answer = np.concatenate((test_s_answer,test_b_answer))
    if weight != -1:
        test_weight = np.concatenate((test_w_s,test_w_b))
        test_data, test_answer, test_weight = shuffle2(test_data, test_answer, test_weight)
    else:
        test_data, test_answer = shuffle2(test_data, test_answer)

    h5test = h5py.File(output_folder+'testing.h5', 'w')
    h5test.create_dataset('testing data', data=test_data)
    h5test.create_dataset('testing answer', data=test_answer)
    if weight != -1:
        h5test.create_dataset('weight', data=test_weight)
    h5test.close()
    output_info.append(["       Saved %s signal and %s background testing events to %s" % (len(test_s),len(test_b),output_folder+"testing.h5")])
    ende = time.time()
    output_info.append(["---------------------------------------------------------------------------------"])
    output_info.append(["   Step 5e finished:  took %s." % (ende-starte)])
    output_info.append(["---------------------------------------------------------------------------------"])
    print_index = print_output(output_info,print_index,print_folder) 
    startf = time.time()
    output_info.append(["   Step 5f:  Saving training data..."])
    output_info.append(["---------------------------------------------------------------------------------"])
    for k in range(num_subsets):
        output_info.append(["       Subset %s:" % k])
        start_s = k*int(len(train_s)/num_subsets)
        end_s = (k+1)*int(len(train_s)/num_subsets)
        start_b = k*int(len(train_b)/num_subsets)
        end_b = (k+1)*int(len(train_b)/num_subsets)
        output_info.append(["           Signal     start:   %s" % start_s])
        output_info.append(["           Signal       end:   %s" % end_s])
        output_info.append(["           Background start:   %s" % start_b])
        output_info.append(["           Background   end:   %s" % end_b])
        if k == 0:
            temp_train_s = train_s[:end_s]
            temp_train_b = train_b[:end_b]
        elif k == num_subsets-1:
            temp_train_s = train_s[start_s:]
            temp_train_b = train_b[start_b:]
        else:
            temp_train_s = train_s[start_s:end_s]
            temp_train_b = train_b[start_b:end_b]
        if weight != -1:
            if k == 0:
                temp_train_w_s = train_w_s[:end_s]
                temp_train_w_b = train_w_b[:end_b]
            elif k == num_subsets-1:
                temp_train_w_s = train_w_s[start_s:]
                temp_train_w_b = train_w_b[start_b:]
            else:
                temp_train_w_s = train_w_s[start_s:end_s]
                temp_train_w_b = train_w_b[start_b:end_b]
        temp_s_answer = np.asarray([1.0 for i in range(len(temp_train_s))])
        temp_b_answer = np.asarray([0.0 for i in range(len(temp_train_b))])

        temp_train = np.concatenate((temp_train_s,temp_train_b))
        temp_answer = np.concatenate((temp_s_answer,temp_b_answer))
        if weight != -1:
            temp_weight = np.concatenate((temp_train_w_s,temp_train_w_b))
            temp_train, temp_answer, temp_weight = shuffle2(temp_train, temp_answer, temp_weight)
        else:
            temp_train, temp_answer = shuffle2(temp_train, temp_answer)

        h5temp_train = h5py.File(output_folder+'training%s.h5' % k, 'w')
        h5temp_train.create_dataset('training data', data=temp_train)
        h5temp_train.create_dataset('training answer', data=temp_answer)
        if weight != -1:
            h5temp_train.create_dataset('weight', data=temp_weight)
        h5temp_train.close()
        output_info.append(["       Saved %s signal and %s background training events to %s" % (len(temp_train_s),len(temp_train_b),output_folder+"training%s.h5" % k)])
        print_index = print_output(output_info,print_index,print_folder) 
    endf = time.time()
    output_info.append(["---------------------------------------------------------------------------------"])
    output_info.append(["   Step 5f finished:  took %s." % (endf-startf)])
    output_info.append(["---------------------------------------------------------------------------------"])
    preprocessing_end = time.time()
    output_info.append(["    Entire Job Finished:   took %s." % (preprocessing_end-preprocessing_start)])
    output_info.append(["---------------------------------------------------------------------------------"])
    print_index = print_output(output_info,print_index,print_folder)     
    output_txt = ''
    for i in range(len(output_info)):
        output_txt += output_info[i][0]
        output_txt += "\n"
    with open(output_folder+"output.txt","w") as file:
        file.write(output_txt)   
    return output_info
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
# function for generating and running a LUX model using subsets
#---------------------------------------------------------------------------------------------------------------
def generate_model(
    exp_folder: str,            # location to store results
    data_folder: str,           # location of the data
    process_folder: str,        
    model_folder: str,          # specific folder for model
    signal_file: str,           # signal file to upload
    background_file: str,       # background file to upload
    var_set: List[int]=[],      # list of variable indices
    weight: int=-1,             # location of any weights in the files
    test_split: float=.2,       # amount of testing to set aside
    num_subsets: int=1,         # number of subsets to split training into
    mi_thres: float=.05,        # threshold to be within to break out of training
    mi_input_val: float=-1,     # possible input value of mi maximum
    topology: List[int]=[],     # topology to use for the network
    activations: List[str]=[],  # activations to use after each layer
    epochs: int=10,             # number of epochs in an iteration
    num_iterations: int=10,     # number of iterations to use
    batch_size: int=100,        # batch size to use for training
    use_auc: bool=False,        # use AUC metric during training
    load_model: str='',         # location to load existing model from
    normalized: bool=False,     # whether the data is normalized or not
    balanced: bool=False,       # whether the weights are balanced or not
    shuffle: bool=True,         # whether to shuffle the events used before training
    plot_history: bool=True,    # to plot history
    plot_auc: bool=True,        # to plot auc
    save_tfpr: bool=False,      # whether or not to save tpr and fprs
    preprocess: bool=True       # whether to preprocess the data
) -> None: 
    model_start = time.time()
    #   first we generate the folder
    make_folder(exp_folder + model_folder)
    # keep track of what output_info we are printing
    print_index, print_folder = 0, exp_folder+model_folder+"generate_model_output.csv"
    # we want to store output information into a txt file
    output_info = [["---------------------------------------------------------------------------------"],
                   ["                    Neural Network Model Generation with Subsets                 "],
                   ["---------------------------------------------------------------------------------"],
                   [" Parameters:                                   "],
                   ["   a) exp_folder:         %s" % exp_folder],
                   ["   b) data_folder:        %s" % data_folder],
                   ["   c) model_folder:       %s" % model_folder],
                   ["   d) signal_file:        %s" % signal_file],
                   ["   e) background_file:    %s" % background_file],
                   ["   f) var_set:            %s" % var_set],
                   ["   g) weight:             %s" % weight],
                   ["   h) test_split:         %s" % test_split],
                   ["   i) num_subsets:        %s" % num_subsets],
                   ["   j) mi_thres:           %s" % mi_thres],
                   ["   k) mi_input_val:       %s" % mi_input_val],
                   ["   l) topology:           %s" % topology],
                   ["   m) activations:        %s" % activations],
                   ["   n) epochs:             %s" % epochs],
                   ["   o) num_iterations:     %s" % num_iterations],
                   ["   p) batch_size:         %s" % batch_size],
                   ["   q) use_auc:            %s" % use_auc],
                   ["   r) load_model:         %s" % load_model],
                   ["   s) normalized:         %s" % normalized],
                   ["   t) balanced:           %s" % balanced],
                   ["   u) shuffle:            %s" % shuffle],
                   ["   v) plot_history:       %s" % plot_history],
                   ["   w) plot_auc:           %s" % plot_auc],
                   ["   x) save_tfpr:          %s" % save_tfpr],
                   ["   y) preprocess_data:    %s" % preprocess]]
    print_index = print_output(output_info,print_index,print_folder)
    output_info += get_system_info()
    print_index = print_output(output_info,print_index,print_folder)
    # load the data
    ###########################################################################
    # Step 1
    start = time.time()
    output_info.append([" Step 1:  Preprocessing data..."])
    if preprocess:
        output_info.append(preprocess_data(
            data_folder,
            signal_file,
            background_file,
            data_folder+process_folder,
            var_set,
            weight,
            test_split,
            normalized,
            balanced,
            num_subsets
        ))
    end = time.time()
    output_info.append(["---------------------------------------------------------------------------------"])
    output_info.append(["    Step 1 finished:  took %s." % (end-start)])
    output_info.append(["---------------------------------------------------------------------------------"])
    start = time.time()
    output_info.append([" Step 2:  Setting up model..."])
    output_info.append(["---------------------------------------------------------------------------------"])
    # set up the models
    output_info.append(["   Setting up model with topology: %s," % topology])
    output_info.append(["                 number of epochs: %s," % epochs])
    output_info.append(["                   and batch size: %s." % batch_size])
    # create the model
    mlp = MLP(topology,optimizer='Nadam',activation=activations,use_auc=use_auc)
    # add normalization params from before
    mlp.set_norm_params_from_file(data_folder+process_folder+"norm_params.csv")
    if load_model != '':
        output_info.append(["   Loading model from file: %s" % load_model])
        mlp.set_weights_from_file(load_model+"weights_after.csv",load_model+"biases_after.csv")
        output_info.append(["   Model successfully loaded from file."])
    output_info.append(["             number of parameters: %s." % mlp.model.count_params()])
    for key, value in mlp.model.optimizer.get_config().items():
        output_info.append(["{:>33}: {}".format(key,value)])
    output_info.append(["---------------------------------------------------------------------------------"])
    end = time.time()
    output_info.append([" Step 2 finished:   took %s." % (end-start)])
    output_info.append(["---------------------------------------------------------------------------------"])
    print_index = print_output(output_info,print_index,print_folder)
    ###########################################################################
    # Step 5
    start = time.time()
    output_info.append([" Step 3:  Training model..."])
    output_info.append(["---------------------------------------------------------------------------------"])
    print_index = print_output(output_info,print_index,print_folder)
    current_iter = 0
    # load in test data
    test_file = h5py.File(data_folder+process_folder+'testing.h5', 'r')
    test_data = np.array(test_file.get('testing data'))
    test_answer = np.array(test_file.get('testing answer'))
    leakages = []
    mis = []
    aucs = []
    train_leakages, train_aucs, train_mis = [], [], []
    training_stats = [['epoch','mi','auc','leakage']]
    validation_stats = [['epoch','mi','auc','leakage']]
    if weight != -1:
        test_weights = np.array(test_file.get('weight'))
    curr_epoch = 0
    while current_iter < num_iterations:
        #output_info.append(["| {:^102} |".format("-----------------------------------------------------------")])
        temp_mis, temp_aucs, temp_leakages = [], [], []
        if weight != -1:
            temp_mis, temp_aucs, temp_leakages = mlp.train_file_subsets(
                data_folder+process_folder,
                num_epochs=epochs,
                batch=batch_size,
                weight=1,
                num_subsets=num_subsets,
                shuffle=shuffle,
                verbose=1,
                metrics=True
            )
        else:
            temp_mis, temp_aucs, temp_leakages = mlp.train_file_subsets(
                data_folder+process_folder,
                num_epochs=epochs,
                batch=batch_size,
                num_subsets=num_subsets,
                shuffle=shuffle,
                verbose=1,
                metrics=True
            )
        for i in range(len(temp_mis)):
            training_stats.append([curr_epoch,temp_mis[i][0],temp_aucs[i],temp_leakages[i]])
            #train_mis.append(temp_mis[i])
            #train_aucs.append(temp_aucs[i])
            #train_leakages.append(temp_leakages[i])
            curr_epoch += 1
        print_index = print_output(output_info,print_index,print_folder)
        #output_info.append(["| {:^102} |".format("-----------------------------------------------------------")]) 
        output_info.append(["| {:^102} |".format("Iteration "+str(current_iter+1)+" of "+str(num_iterations))])
        output_info.append(["| {:^7} | {:^7} | {:^9} | {:^15} | {:^12} | {:^19} | {:^15} |".format("Epoch","Subset","Time","Binary Accuracy","Loss","Validation Accuracy","Validation Loss")])
        for l in range(epochs):
            for k in range(num_subsets):
                kk = k + num_subsets * current_iter
                output_info.append(["| {:^7s} | {:^7s} | {:^9.2e} | {:^15.4e} | {:^12.4e} | {:^19.4e} | {:^15.4e} |".format(str(l+1)+"/"+str(epochs),
                                                                                                                            str(k+1)+"/"+str(num_subsets),
                                                                                                                            mlp.time_callback.times[l],
                                                                                                                            mlp.history[kk].history['binary_accuracy'][l],
                                                                                                                            mlp.history[kk].history['loss'][l],
                                                                                                                            mlp.history[kk].history['val_binary_accuracy'][l],
                                                                                                                            mlp.history[kk].history['val_loss'][l])])
        current_iter += 1
        startmi = time.time()
        output_info.append(["---------------------------------------------------------------------------------"])
        output_info.append(["    Step 5a:  Computing current model MI..."])
        output_info.append(["---------------------------------------------------------------------------------"])
        print_index = print_output(output_info,print_index,print_folder)
        results = mlp.evaluate(test_data,test_answer)
        temp_s = [[results[i][0]] for i in range(len(results)) if results[i][1] == 1]
        temp_b = [[results[i][0]] for i in range(len(results)) if results[i][1] != 1]
        if weight != -1:
            w_s = [test_weights[i] for i in range(len(test_weights)) if results[i][1] == 1]
            w_b = [test_weights[i] for i in range(len(test_weights)) if results[i][1] != 1]
        current_mi = 0.0
        if weight != -1:
            current_mi, ns, nb = mi.mi_binary_weights(temp_s,temp_b,w_s,w_b,k=1)
            output_info.append(["      Number of signal events used:     %s," % ns])
            output_info.append(["      Number of background events used: %s," % nb])
            output_info.append(["      Binary weighted MI after:        %s." % current_mi])
        else:
            current_mi = mi.mi_binary(temp_s,temp_b,k=1)
            output_info.append(["      Binary MI after: %s." % current_mi])
        endmi = time.time()
        output_info.append(["---------------------------------------------------------------------------------"])
        output_info.append([" Step 5a finished:   took %s." % (endmi-startmi)])
        output_info.append(["---------------------------------------------------------------------------------"])
        startauc = time.time()
        output_info.append([" Step 5b:  Determining AUC,leakage"])
        output_info.append(["---------------------------------------------------------------------------------"])
        fpr_keras, tpr_keras, thresholds_keras, auc_keras = mlp.plot_weighted_auc(test_data,test_answer,test_weights,show=False,normalized=True)
        leakage = fpr_keras[bisect.bisect_left(tpr_keras,0.5)]
        output_info.append(["       AUC:                %s" % auc_keras])
        output_info.append(["       Leakage (50 per.):  %s" % leakage])
        output_info.append(["---------------------------------------------------------------------------------"])
        endauc = time.time()
        output_info.append(["   Step 5b finished: took %s." % (endauc - startauc)])
        output_info.append(["---------------------------------------------------------------------------------"])
        startmi = time.time()
        output_info.append(["    Step 5b:  Determining if threshold met..."])
        output_info.append(["---------------------------------------------------------------------------------"])
        print_index = print_output(output_info,print_index,print_folder)
        if (current_mi > (mi_input_val * (1 - mi_thres))):
            output_info.append(["  MI threshold value of {} met!  Sufficiency: {}".format(mi_thres,str(current_mi/mi_input_val))])
            current_iter = num_iterations
        else:
            output_info.append(["  MI threshold value of {} not met!  Sufficiency: {}".format(mi_thres,str(current_mi/mi_input_val))])
        endmi = time.time()
        output_info.append(["---------------------------------------------------------------------------------"])
        output_info.append([" Step 5b finished:   took %s." % (endmi-startmi)])
        output_info.append(["---------------------------------------------------------------------------------"])
        print_index = print_output(output_info,print_index,print_folder)
        leakages.append(leakage)
        aucs.append(auc_keras)
        mis.append(current_mi)
        validation_stats.append([curr_epoch-1,current_mi,auc_keras,leakage])
    ###########################################################################
    # Step 6
    # saving model
    # saving mi, auc and leakage stats
    with open(exp_folder+model_folder+'validation_stats.csv',"w") as file:
        writer = csv.writer(file,delimiter=",")
        #writer.writerows([mis,aucs,leakages])
        writer.writerows(validation_stats)
    start = time.time()
    output_info.append([" Step 6:  Saving model..."])
    output_info.append(["---------------------------------------------------------------------------------"])
    mlp.save_weights_to_file(exp_folder+model_folder+'weights_after.csv')
    mlp.save_biases_to_file(exp_folder+model_folder+'biases_after.csv')
    mlp.save_model(exp_folder+model_folder+'model_after.csv')
    end = time.time()

    output_info.append([" Step 6 finished:   took %s." % (end-start)])
    output_info.append(["---------------------------------------------------------------------------------"])
    print_index = print_output(output_info,print_index,print_folder)
    ###########################################################################
    # Step 7
    start = time.time()
    output_info.append([" Step 7:  Evaluating model..."])
    output_info.append(["---------------------------------------------------------------------------------"])
    print_index = print_output(output_info,print_index,print_folder)
    results = mlp.evaluate(test_data,test_answer)
    temp_s = [[results[i][0]] for i in range(len(results)) if results[i][1] == 1]
    temp_b = [[results[i][0]] for i in range(len(results)) if results[i][1] != 1]
    if weight != -1:
        w_s = [test_weights[i] for i in range(len(test_weights)) if results[i][1] == 1]
        w_b = [test_weights[i] for i in range(len(test_weights)) if results[i][1] != 1]
    ###########################################################################
    # Step 7b
    startb = time.time()
    output_info.append(["    Step 7b:  Plotting History..."])
    output_info.append(["---------------------------------------------------------------------------------"])
    if plot_history:
        mlp.plot_history(save=True,show=False,filename=exp_folder+model_folder+'history')
    endb = time.time()
    output_info.append(["    Step 7b finished: took %s." % (endb-startb)])
    output_info.append(["---------------------------------------------------------------------------------"])
    print_index = print_output(output_info,print_index,print_folder)
    ###########################################################################
    # Step 7c
    startc = time.time()
    output_info.append(["    Step 7c: Determing AUC, "])
    output_info.append(["             leakage and threshold,"])
    output_info.append(["             and plotting ROC..."])
    output_info.append(["---------------------------------------------------------------------------------"])
    print_index = print_output(output_info,print_index,print_folder)
    if weight != -1:
        fpr_keras, tpr_keras, thresholds_keras, auc_keras = mlp.plot_weighted_auc(test_data,
                                                                                  test_answer,
                                                                                  test_weights,
                                                                                  show=False,
                                                                                  filename=exp_folder+model_folder+'roc',
                                                                                  normalized=True)
    else:
        fpr_keras, tpr_keras, thresholds_keras, auc_keras = mlp.plot_auc(test_data,
                                                                         test_answer,
                                                                         show=False,
                                                                         filename=exp_folder+model_folder+'roc',
                                                                         normalized=True)   
    if save_tfpr:
        output_info.append(["   Saving true/false positive and threshold info..."])                                                                     
        with open(exp_folder+model_folder+'fpr_values.csv','w') as file:
            writer = csv.writer(file,delimiter=",")
            writer.writerows([fpr_keras])
        with open(exp_folder+model_folder+'tpr_values.csv','w') as file:
            writer = csv.writer(file,delimiter=",")
            writer.writerows([tpr_keras])
        with open(exp_folder+model_folder+'threshold_values.csv','w') as file:
            writer = csv.writer(file,delimiter=",")
            writer.writerows([thresholds_keras])
    endc = time.time()
    leakage = fpr_keras[bisect.bisect_left(tpr_keras,0.5)]      # bkg leakage at 50% signal efficiency
    thres = thresholds_keras[bisect.bisect_left(tpr_keras,0.5)] # signal threshold at 50% signal efficiency
    output_info.append(["    Stats:"])
    output_info.append(["      AUC:                  %s" % auc_keras])
    output_info.append(["      Leakage (50 per.):    %s" % leakage])
    output_info.append(["      Threshold (50 per.):  %s" % thres])
    output_info.append(["---------------------------------------------------------------------------------"])
    output_info.append(["    Step 7c finished: took %s." % (endc-startc)])
    output_info.append(["---------------------------------------------------------------------------------"])
    print_index = print_output(output_info,print_index,print_folder)
    ###########################################################################
    # Step 7d
    startd = time.time()
    output_info.append(["    Step 7d:  Computing output MI..."])
    output_info.append(["---------------------------------------------------------------------------------"])
    print_index = print_output(output_info,print_index,print_folder)
    if weight != -1:
        mi_after, ns, nb = mi.mi_binary_weights(temp_s,temp_b,w_s,w_b,k=3)
        output_info.append(["      Number of signal events used:     %s," % ns])
        output_info.append(["      Number of background events used: %s," % nb])
        output_info.append(["      Binary weighted MI after:        %s." % mi_after])
    else:
        mi_after = mi.mi_binary(temp_s,temp_b,k=3)
        output_info.append(["      Binary MI after: %s." % mi_after])
    endd = time.time()

    output_info.append(["---------------------------------------------------------------------------------"])
    output_info.append(["    Step 7d finished: took %s." % (endd-startd)])
    output_info.append(["---------------------------------------------------------------------------------"])
    print_index = print_output(output_info,print_index,print_folder)
    ###########################################################################
    # Step 7e
    output_info.append(["    Step 7e:  Saving stats and output..."])
    output_info.append(["---------------------------------------------------------------------------------"])
    print_index = print_output(output_info,print_index,print_folder)
    with open(exp_folder+model_folder+"stats.csv","w") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerows([['mi_input_val','mi_after','auc','leakage','threshold'],[mi_input_val,mi_after,auc_keras,leakage,thres]])
    with open(exp_folder+model_folder+"training_stats.csv","w") as file:
        writer = csv.writer(file,delimiter=",")
        writer.writerows(training_stats)
    end = time.time()
    output_info.append([" Step 7 finished:   took %s." % (end-start)])
    output_info.append(["---------------------------------------------------------------------------------"])
    print_index = print_output(output_info,print_index,print_folder)
    model_end = time.time()
    output_info.append(["    Entire Job Finished:   took %s." % (model_end-model_start)])
    output_info.append(["---------------------------------------------------------------------------------"])
    print_index = print_output(output_info,print_index,print_folder)
    with open(exp_folder+model_folder+"output.txt","w") as file:
        writer = csv.writer(file,delimiter=",")
        writer.writerows(output_info)
    #output_txt = ''
    #for i in range(len(output_info)):
    #    output_txt += output_info[i][0]
    #    output_txt += "\n"
    #with open(exp_folder+model_folder+"output.txt","w") as file:
    #    file.write(output_txt)   
#---------------------------------------------------------------------------------------------------------------