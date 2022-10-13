"""
"""
import os
import numpy as np
import uproot
import time
import psutil
import platform
import GPUtil
import datetime
from datetime import date, datetime
from typing import List
from sklearn.utils import shuffle as shuffle2

# function for making a folder
def make_folder(
    name: str
) -> None:
    if not os.path.exists(name):
        os.makedirs(name)

#-----------------------------------------------------------------------------
#   Converting csv and root files to npz
#-----------------------------------------------------------------------------
#   convert_csv_to_npz(file       - specify the filename "file"
#                      var_set    - the column numbers of the variables
#                      class_list - possible list of different classes
#                      class_loc  - possible column for the class label)
#-----------------------------------------------------------------------------
def convert_csv_to_npz(
    file: str,
    var_set: list,
    class_list=[],
    class_loc=-1
):
    # check that csv file exists
    if(not os.path.exists(file+".csv")):
        print("File " + file + ".csv does not exist!")
        return
    data = []
    with open(file+".csv",'r') as temp:
        reader = csv.reader(temp,delimiter=",")
        for row in reader:
            data.append([float(row[i]) for i in var_set])
    # if there are no classes
    if(class_loc == -1):
        np.savez(file+'.npz',np.asarray(data,dtype=np.float32))
        print("Set of %s events in %s variables saved in file %s.npz" %
              (len(data),len(var_set),file))
    else:
        for j in range(len(class_list)):
            temp_data = [data[i] for i in range(len(data)) if
                         data[i][class_loc] == class_list[j]]
            np.savez(file+'_class%s.npz' % j,np.asarray(temp_data,
                                                        dtype=np.float32))
            print("Set of %s events in %s variables saved in %s_class%s.npz"
                  % (len(data),len(var_set),file,j))
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#   convert_root_to_npz(file       - specify the filename "file"
#                       tree       - name of the tree
#                       var_set    - list of variable names in the tree
#-----------------------------------------------------------------------------
def convert_root_to_npz(
    file: str,
    tree: str,
    var_set: list
):
    # check that csv file exists
    if(not os.path.exists(file+".root")):
        print("File " + file + ".root does not exist!")
        return
    rootfile = uproot.open(file+'.root')
    temp = []
    for j in range(len(var_set)):
        temp.append(rootfile[tree][var_set[j]].array(library="np"))
    data = [[temp[i][j] for i in range(len(var_set))]
            for j in range(len(temp[0]))]
    np.savez(file+'.npz',data)
    print("Set of %s events in %s variables saved in file %s.npz" %
              (len(data),len(var_set),file))
#-----------------------------------------------------------------------------

def balance_weights(weights, answer):
    sig_weights = [weights[i] for i in range(len(weights)) if answer[i][0] > 0]
    back_weights = [weights[i] for i in range(len(weights)) if answer[i][0] < 0]
    sig_max, back_max = max(sig_weights), max(back_weights)
    sig_weights = [sig_weights[i]/sig_max for i in range(len(sig_weights))]
    back_weights = [back_weights[i]/back_max for i in range(len(back_weights))]
    fraction = sum(sig_weights)/sum(back_weights)
    output_string = [["   Sum of old weights = %s" % sum(weights)]]
    if fraction >= 1.0:
        weight_s = 1.0/fraction
        weight_b = 1.0
        output_string.append(["   Ratio of signal to background weights = %s" % weight_s])
    else:
        weight_s = 1.0
        weight_b = fraction
        output_string.append(["   Ratio of background to signal weights = %s" % weight_b])
    new_weights = []
    for i in range(len(weights)):
        if answer[i][0] > 0:
            new_weights.append(weights[i]*weight_s/sig_max)
        else:
            new_weights.append(weights[i]*weight_b/back_max)
    output_string.append(["   Sum of re-weighted samples = %s" % sum(new_weights)])
    return new_weights, output_string
#-----------------------------------------------------------------------------
#   generate_binary_training_testing_data(signal_file     - the signal file
#                                         background_file - the background file
#                                         labels          - the labels in npz
#                                         var_set      - which variables to use
#                                         ratio_of_file- amount of file to use
#                                         testing_ratio- amount of data testing
#                                         symmetric_signals- p(s) = p(b)
#                                         percentage_signal- p(s)
#                                         percentage_background- p(b))
#                                         weights is the location of the weights
#-----------------------------------------------------------------------------
def generate_binary_training_testing_data(
    signal_file: str,
    background_file: str,
    labels: list,
    var_set=[],
    ratio_of_file=1.0,
    testing_ratio=0.3,
    symmetric_signals=False,
    percentage_signal=1.0,
    percentage_background=1.0,
    init_shuffle=False,
    weight=-1
):
    # First load up the signal and background files, then determine the
    # amount of the files to use with ratio_of_file
    #print("loading...")
    load_start_time = time.time()
    signal = np.load(signal_file)[labels[0]]
    background = np.load(background_file)[labels[1]]
    #   if the user only wants certain variables
    #   if the user has weights,
    load_end_time = time.time()
    
    # if theres a weight, add it to the var_set
    if weight != -1:
        var_set.append(weight)
    # if var_set is specified, use it to cull the inputs
    if (var_set != []):
        signal = signal[:,var_set]
        background = background[:,var_set]
    shuffle_start_time = time.time()
    if init_shuffle:
        np.random.shuffle(signal)
        np.random.shuffle(background)
    shuffle_end_time = time.time()
    # Now determine which is smallest, the length of signal background
    # or the ratios
    temp_data = []
    data, answer = [], []
    sym_start_time = time.time()
    if (symmetric_signals==True):
        num_of_events = np.amin([len(signal),len(background),
                                 int(len(signal)*ratio_of_file),
                                 int(len(background)*ratio_of_file)])
        num_signal = num_of_events
        num_background = num_of_events
    else:
        num_signal = int(percentage_signal * len(signal))
        num_background = int(percentage_background * len(background))
    for j in range(num_signal):
#       temp_data.append([signal[j],[1.0]])
        answer.append([1.0])
        data.append(signal[j])
    for j in range(num_background):
#       temp_data.append([background[j],[-1.0]])
        answer.append([-1.0])
        data.append(background[j])
    data = np.asarray(data)
    answer = np.asarray(answer)
    data, answer = shuffle2(data,answer)
#   data = temp_data[:,0]
#   answer = temp_data[:,1]
    sym_end_time = time.time()
    
    weights_sep_start_time = time.time()
    if weight != -1:
        weights = data[:,-1]    
        data = data[:,:-1]
#       weights = [data[i][-1] for i in range(len(data))]
#       data = [[data[i][j] for j in range(len(data[0])-1)] for i in range(len(data))]
    weights_sep_end_time = time.time()
    
    delta_load = (load_end_time - load_start_time)
    delta_shuf = (shuffle_end_time - shuffle_start_time)
    delta_sym  = (sym_end_time - sym_start_time)
    delta_weig = (weights_sep_end_time - weights_sep_start_time)
#   print("loading time:            %s" % delta_load)
#   print("init shuffle time:       %s" % delta_shuf)
#   print("symmetrization time:     %s" % delta_sym)
#   print("separating weights time: %s" % delta_weig)
    if ( testing_ratio == 0.0 ):
#       print("total time to load data: %s" % (delta_load+delta_shuf+delta_sym+delta_weig))
#       print("Loaded files %s and %s with %s events for training and 0 events \
#              for testing." % (signal_file, background_file, len(data)))
        if weight != -1:
            return data, answer, weights
        else:
            return data, answer
    else:
        if weight != -1:
            #   Otherwise we partition the amount of testing data
            part_start_time = time.time()
            num_of_testing = int(len(data)*testing_ratio)
            train_data = list( data[:][:-num_of_testing] )
            train_answer = list( answer[:][:-num_of_testing] )
            test_data = list( data[:][-num_of_testing:] )
            test_answer = list( answer[:][-num_of_testing:] )
            train_weight = list( weights[:-num_of_testing] )
            test_weight = list( weight[-num_of_testing:] )
            part_end_time = time.time()
            delta_part = (part_end_time - part_start_time)
            print("partitioning time:      %s" % delta_part)
            print("total time to load data: %s" % (delta_load+delta_shuf+delta_sym+delta_weig+delta_part))
            return train_data, train_answer, test_data, test_answer, train_weight, test_weight
        else:
            part_start_time = time.time()
            num_of_testing = int(len(data)*testing_ratio)
            train_data = list( data[:][:-num_of_testing] )
            train_answer = list( answer[:][:-num_of_testing] )
            test_data = list( data[:][-num_of_testing:] )
            test_answer = list( answer[:][-num_of_testing:] )
            part_end_time = time.time()
            delta_part = (part_end_time - part_start_time)
            print("partitioning time:      %s" % delta_part)
            print("total time to load data: %s" % (delta_load+delta_shuf+delta_sym+delta_weig+delta_part))
            return train_data, train_answer, test_data, test_answer
#-----------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
# get_size
#---------------------------------------------------------------------------------------------------------------
def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
# helper function for printing output
#---------------------------------------------------------------------------------------------------------------
def print_output(
    output: List[List[str]],
    start_index: int,
    outfile: str="temp_output.csv"
) -> int:
    if (start_index == 0):
        with open(outfile,"w") as file:
            for i in range(start_index,len(output)):
                print(output[i][0])
                #file.write(output[i][0])
                #file.write("\n")
    else:
        with open(outfile,"a") as file:
            for i in range(start_index,len(output)):
                print(output[i][0])
                #file.write(output[i][0])
                #file.write("\n")
    return len(output)
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
# get system info
#---------------------------------------------------------------------------------------------------------------
def get_system_info():
    uname = platform.uname()
    cpufreq = psutil.cpu_freq()
    svmem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    gpus = GPUtil.getGPUs()
    today = date.today()
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    system_info = [["---------------------------------------------------------------------------------"],
                   ["                                    System Info                                  "],
                   ["---------------------------------------------------------------------------------"],
                   ["   Date/Time:          {}".format(dt_string)],
                   ["   System:             {}".format(uname.system)],
                   ["   Node Name:          {}".format(uname.node)],
                   ["   Release:            {}".format(uname.release)],
                   ["   Version:            {}".format(uname.version)],
                   ["   Machine:            {}".format(uname.machine)],
                   ["   Processor:          {}".format(uname.processor)],
                   ["---------------------------------------------------------------------------------"],
                   ["                                     CPU Info                                    "],
                   ["---------------------------------------------------------------------------------"],
                   ["   Physical Cores:          {}".format(psutil.cpu_count(logical=False))],
                   ["   Total Cores:             {}".format(psutil.cpu_count(logical=True))],
                   ["   Max Frequency (Mhz):     {:.2f}".format(cpufreq.max)],
                   ["   Min Frequency (Mhz):     {:.2f}".format(cpufreq.min)],
                   ["   Current Frequency (Mhz): {:.2f}".format(cpufreq.current)],
                   ["   CPU Usage Per Core:"]]
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        system_info.append(["       Core {}: {}%".format(i,percentage)])
    system_info.append(["   Total CPU Usage: {}%".format(psutil.cpu_percent())])
    system_info.append(["---------------------------------------------------------------------------------"])
    system_info.append(["                                     Memory Info                                 "])
    system_info.append(["---------------------------------------------------------------------------------"])
    system_info.append(["   Total:          {}".format(get_size(svmem.total))])
    system_info.append(["   Available:      {}".format(get_size(svmem.available))])
    system_info.append(["   Used:           {}".format(get_size(svmem.used))])
    system_info.append(["   Percentage:     {}".format(svmem.percent)])
    system_info.append(["---------------------------------------------------------------------------------"])
    system_info.append(["                                      SWAP Info                                  "])
    system_info.append(["---------------------------------------------------------------------------------"])
    system_info.append(["   Total:          {}".format(get_size(swap.total))])
    system_info.append(["   Free:           {}".format(get_size(swap.free))])
    system_info.append(["   Used:           {}".format(get_size(swap.used))])
    system_info.append(["   Percentage:     {}".format(swap.percent)])
    system_info.append(["---------------------------------------------------------------------------------"])
    system_info.append(["                                      GPU Info                                   "])
    system_info.append(["---------------------------------------------------------------------------------"])
    for gpu in gpus:
        system_info.append(["   id:             {}".format(gpu.id)])
        system_info.append(["   name:           {}".format(gpu.name)])
        system_info.append(["   load:           {}%".format(gpu.load*100)])
        system_info.append(["   free memory:    {}MB".format(gpu.memoryFree)])
        system_info.append(["   used memory:    {}MB".format(gpu.memoryUsed)])
        system_info.append(["   total memory:   {}MB".format(gpu.memoryTotal)])
        system_info.append(["   temperature:    {}°C".format(gpu.temperature)])
        system_info.append(["   uuid:           {}".format(gpu.uuid)])
    return system_info
#---------------------------------------------------------------------------------------------------------------