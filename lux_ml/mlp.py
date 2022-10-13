# ML and information theory scripts                                      Nicholas Carrara
'''
    This code was developed for the paper: 
    "Fast and Flexible Analysis of Direct Dark Matter Search Data with Machine Learning"
    by the LUX Collaboration ().

    Main authors are:
        Nicholas Carrara [nmcarrara@ucdavis.edu]
        University of California at Davis
		Davis, CA 95616
        Scott Kravitz [swkravitz@lbl.gov]
        Lawrence Berkeley National Lab
        Berkeley, CA 94720
'''
#-----------------------------------------------------------------------------
#   Required packages
import os
# set the backend to tensorflow
os.environ['KERAS_BACKEND']='tensorflow'
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from math import log
import csv

from tensorflow.keras import backend as K
from tensorflow.keras import optimizers, models, layers
from tensorflow.keras import initializers as ki

import tensorflow as tf
from sklearn.metrics import auc, roc_curve
import time
import h5py
import bisect

import lux_ml.mi as mi

#-----------------------------------------------------------------------------
# Timing callback
#-----------------------------------------------------------------------------
class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
    def on_train_begin(self, logs={}):
        return
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
#   Neural Network class
#-----------------------------------------------------------------------------
'''
    This class generates a sequential MLP model using the Keras library.
    The user must specify the topology of the network; i.e. the number
    of nodes in each layer; e.g. [4,10,3,1]
    Default values are chosen for the other options, however the user has
    the ability to pick an optimizer, an activation function for each layer,
    and an initializer for the weights.


'''
#-----------------------------------------------------------------------------
class MLP:
    #-------------------------------------------------------------------------
    #   __init__(self,topology - layer structure for the network, [4,10,3,1]
    #                 optimizer- select between; 'SGD' - Stochastic Grad. Des.
    #                                            'RMSprop' - RMS Prop.
    #                                            'Adagrad' - Adagrad
    #                                            'Adadelta'- Adadelta
    #                                            'Adam'    - Adam
    #                                            'Adamax'  - Adamax
    #                                            'Nadam'   - Nadam
    #                 opt_params- 'SGD'     - [lr,momentum,decay,nesterov]
    #                             'RMSprop' - [lr,rho,epsilon,decay]
    #                             'Adagrad' - [lr,epsilon,decay]
    #                             'Adadelta'- [lr,rho,epsilon,decay]
    #                             'Adam'    - [lr,beta_1,beta_2,epsilon,decay,
    #                                          amsgrad]
    #                             'Adamax'  - [lr,beta_1,beta_2,epsilon,decay]
    #                             'Nadam'   - [lr,beta_1,beta_2,epsilon,
    #                                          scheduled_decay]
    #                 activation- select between; 'tanh'     - Tanh function
    #                                             'elu'      - exp. linear
    #                                             'selu'- scaled exp. linear
    #                                             'softplus' - Soft plus
    #                                             'softsign' - Soft sign
    #                                             'relu'     - rect. linear
    #                                             'sigmoid'  - Sigmoid
    #                                             'hard_sigmoid' - Hard Sig.
    #                                             'exponential' - exp
    #                                             'linear' - Linear
    #                 act_params- 'elu'       - [alpha]
    #                             'relu'      - [alpha,max_value,threshold]
    #                 initializer- select between; 'zeros' - set all to zero
    #                                              'ones'  - set all to one
    #                                              'constant' - Constant
    #                                              'normal'- random normal
    #                                              'uniform' - random uniform
    #                                              'truncated_normal'
    #                                              'variance_scaling'
    #                                              'orthogonal'
    #                 init_params - 'constant' - [value]
    #                               'normal'   - [mean,stddev,seed]
    #                               'uniform'  - [minval,maxval,seed]
    #                               'truncated_normal'- [mean,stddev,seed]
    #                               'variance_scaling'- [scale,mode,dist,seed]
    #                               'orthogonal'      - [gain,seed]
    #                 filename - if provided, load this model, ignore params
    #                 use_auc - if true, calculate ROC area under curve during training
    #-------------------------------------------------------------------------
    def __init__(self,
        topology: list=[],
        optimizer='SGD',
        opt_params=[],
        activation=None,
        act_params=[],
        initializer=None,
        init_params=[],
        loss: str='mean_squared_error',
        filename=None,
        use_auc=False
    ):
        self.topology = topology
        self.opt_params = []
        self.activations = []
        self.act_params = []
        self.initializers = []
        self.init_params = []
        self.loss = loss
        self.optimizer_name = optimizer
        self.normalization = 'Standard'
        self.normalization_params = []
        self.history = []
        self.train_ints = []
        self.use_auc = use_auc
        self.time_callback = TimeHistory()

        # load model from given file, ignore all else
        if filename is not None:
            self.set_model_from_file(filename)
            return

        # determine the initializers
        if initializer == None:
            self.initializers = [ki.RandomNormal()
                                 for i in range(len(self.topology)-1)]
        elif isinstance(initializer,str):
            if(initializer == 'zeros'):
                self.initializers = [ki.Zeros()
                                     for i in range(len(self.topology)-1)]
            elif(initializer == 'ones'):
                self.initializers = [ki.Ones()
                                     for i in range(len(self.topology)-1)]
            elif(initializer == 'constant'):
                if len(init_params) == 0:
                    self.initializers = [ki.Constant()
                                         for i in range(len(self.topology)-1)]
                else:
                    assert len(init_params) == 1,"Must provide 1 parameter \
                                                  for constant initialization!"
                    self.initializers = [ki.constant(value=init_params[0])
                                     for i in range(len(self.topology)-1)]
            elif(initializer == 'normal'):
                if len(init_params) == 0:
                    self.initializers = [ki.RandomNormal()
                                         for i in range(len(self.topology)-1)]
                else:
                    assert len(init_params) == 3,"Must provide 3 parameters \
                                                  for normal initialization!"
                    self.initializers = [ki.RandomNormal(mean=init_params[0],
                                                   stddev=init_params[1],
                                                   seed=init_params[2])
                                     for i in range(len(self.topology)-1)]
            elif(initializer == 'uniform'):
                if len(init_params) == 0:
                    self.initializers = [ki.RandomUniform()
                                         for i in range(len(self.topology)-1)]
                else:
                    assert len(init_params) == 3,"Must provide 3 parameters \
                                                  for uniform initialization!"
                    self.initializers = [ki.RandomUniform(minval=init_params[0],
                                                    maxval=init_params[1],
                                                    seed=init_params[2])
                                     for i in range(len(self.topology)-1)]
            elif(initializer == 'truncated_normal'):
                if len(init_params) == 0:
                    self.initializers = [ki.TruncatedNormal()
                                         for i in range(len(self.topology)-1)]
                else:
                    assert len(init_params) == 3,"Must provide 3 parameters \
                                                  for truncated normal \
                                                  initialization!"
                    self.initializers = [ki.TruncatedNormal(init_params[0],
                                                             init_params[1],
                                                             init_params[2])
                                     for i in range(len(self.topology)-1)]
            elif(initializer == 'variance_scaling'):
                if len(init_params) == 0:
                    self.initializers = [ki.VarianceScaling()
                                         for i in range(len(self.topology)-1)]
                else:
                    assert len(init_params) == 4,"Must provide 4 parameters \
                                                  for variance scaling \
                                                  initialization!"
                    assert isinstance(opt_params[1],str), "Parameter 2 must \
                                                           be of type string!"
                    assert isinstance(opt_params[2],str), "Parameter 3 must \
                                                           be of type string!"
                    assert (opt_params[1] == 'fan_in' or opt_params[1] ==
                            'fan_out' or opt_params[1] == 'fan_avg'), "Mode \
                            must be either 'fan_in', 'fan_out' or 'fan_avg'!"
                    assert (opt_params[2] == 'normal' or opt_params[2] ==
                            'uniform'), "Distribution must be either 'normal'\
                                         or 'uniform'!"
                    self.initializers = [ki.variance_scaling(init_params[0],
                                                             init_params[1],
                                                             init_params[2],
                                                             init_params[3])
                                     for i in range(len(self.topology)-1)]
            elif(initializer == 'orthogonal'):
                if len(init_params) == 0:
                    self.initializers = [ki.Orthogonal()
                                         for i in range(len(self.topology)-1)]
                else:
                    assert len(init_params) == 2,"Must provide 2 parameters \
                                                  for orthogonal \
                                                  initialization!"
                    self.initializers = [ki.Orthogonal(gain=opt_params[0],
                                                       seed=opt_params[1])
                                     for i in range(len(self.topology)-1)]
        elif isinstance(initializer,list):
            assert len(initializer) == len(self.topology)-1,"Must provide an \
                                                            initializer for \
                                                            each layer!"
            assert len(init_params) == len(self.topology)-1,"Must provide \
                                                            params for \
                                                            each layer \
                                                            initializer!"
            for j in range(len(initializer)):
                if(initializer[j] == 'zeros'):
                    self.initializers.append(ki.Zeros())
                elif(initializer[j] == 'ones'):
                    self.initializers.append(ki.Ones())
                elif(initializer[j] == 'constant'):
                    if len(init_params[j]) == 0:
                        self.initializers.append(ki.Constant())
                    else:
                        assert len(init_params[j]) == 1,"Must provide 1 \
                                                         parameter for \
                                                         constant \
                                                         initialization!"
                        self.initializers.append(ki.Constant(
                                                          init_params[j][0]))
                elif(initializer[j] == 'normal'):
                    if len(init_params[j]) == 0:
                        self.initializers.append(ki.RandomNormal())
                    else:
                        assert len(init_params[j]) == 3,"Must provide 3 \
                                                         parameters for \
                                                         normal \
                                                         initialization!"
                        self.initializers.append(ki.RandomNormal(init_params[j][0],
                                                           init_params[j][1],
                                                           init_params[j][2]))
                elif(initializer[j] == 'uniform'):
                    if len(init_params[j]) == 0:
                        self.initializers.append(ki.Uniform())
                    else:
                        assert len(init_params[j]) == 3,"Must provide 3 \
                                                         parameters for \
                                                         uniform \
                                                         initialization!"
                        self.initializers.append(ki.Uniform(init_params[j][0],
                                                            init_params[j][1],
                                                            init_params[j][2]))
                elif(initializer[j] == 'truncated_normal'):
                    if len(init_params[j]) == 0:
                        self.initializers.append(ki.TruncatedNormal())
                    else:
                        assert len(init_params[j]) == 3,"Must provide 3 \
                                                         parameters for \
                                                         truncated normal\
                                                         initialization!"
                        self.initializers.append(ki.TruncatedNormal(
                                                            init_params[j][0],
                                                            init_params[j][1],
                                                            init_params[j][2]))
                elif(initializer[j] == 'variance_scaling'):
                    if len(init_params[j]) == 0:
                        self.initializers.append(ki.VarianceScaling())
                    else:
                        assert len(init_params[j]) == 4,"Must provide 4 \
                                                         parameters for \
                                                         variance scaling\
                                                         initialization!"
                        assert isinstance(opt_params[1],str),"Parameter 2\
                                                              must be of\
                                                              type string!"
                        assert isinstance(opt_params[2],str),"Parameter 3\
                                                              must be of\
                                                              type string!"
                        assert (opt_params[1] == 'fan_in' or
                                opt_params[1] == 'fan_out' or
                                opt_params[1] == 'fan_avg'),"Mode must be\
                                                             either 'fan_in',\
                                                             'fan_out' \
                                                             or 'fan_avg'!"
                        assert (opt_params[2] == 'normal' or
                                opt_params[2] == 'uniform'),"Distribution\
                                                             must be either\
                                                             'normal' or \
                                                             'uniform'!"
                        self.initializers.append(ki.VarianceScaling(
                                                scale=init_params[j][0],
                                                mode=init_params[j][1],
                                                distribution=init_params[j][2],
                                                seed=init_params[j][3]))
                elif(initializer[j] == 'orthogonal'):
                    if len(init_params[j]) == 0:
                        self.initializers.append(ki.Orthogonal())
                    else:
                        assert len(init_params[j]) == 2,"Must provide 2 \
                                                         parameters for \
                                                         orthogonal \
                                                         initialization!"
                        self.initializers.append(ki.Orthogonal(
                                                    gain=opt_params[0],
                                                    seed=opt_params[1]))

        # set optimizer using optimizer_name
        self.set_optimizer()

        # determine the activations
        if activation == None:
            # set all activations to tanh
            self.activations = ['tanh' for i in range(len(self.topology))]
        elif isinstance(activation,str):
            assert (activation in ['tanh','elu','selu','softplus',
                                   'softsign','relu','sigmoid','hard_sigmoid',
                                   'exponential','linear']), "Activation \
                                   must be one of the allowed types!"
            self.activations = [activation for i in range(len(self.topology))]
            if (activation == 'elu'):
                if len(act_params) == 0:
                    self.act_params = [[1.0] for i in range(len(self.topology))]
                else:
                    assert len(act_params) == 1, "Must provide 1 parameter for\
                                                  elu activation!"
                    self.act_params = [act_params[0] for i in
                                       range(len(self.topology))]
            if (activation == 'relu'):
                if len(act_params) == 0:
                    self.act_params = [[0.0,None,0.0] for i in
                                       range(len(self.topology))]
                else:
                    assert len(act_params) == 3,"Must provide 3 parameters for\
                                                 relu activation!"
                    self.act_params = [act_params for i in
                                       range(len(self.topology))]
        elif isinstance(activation,list):
            assert len(activation) == len(self.topology),"Number of activations\
                                                          must equal the \
                                                          number of layers!"
            for j in range(len(activation)):
                assert (activation[j] in ['tanh','elu','selu',
                                          'softplus','softsign','relu',
                                        'sigmoid','hard_sigmoid','exponential',
                                        'linear']),"Activation must be one of \
                                                    the allowed types!"
                self.activations.append(activation[j])
                if (activation[j] == 'elu'):
                    if len(act_params[j]) == 0:
                        self.act_params.append([1.0])
                    else:
                        assert len(act_params[j]) == 1,"Must provide 1 \
                                                        parameter \
                                                        for elu activation!"
                        self.act_params.append(act_params[0])
                elif (activation[j] == 'relu'):
                    if len(act_params[j]) == 0:
                        self.act_params.append([0.0,None,0.0])
                    else:
                        assert len(act_params[j]) == 3,"Must provide 3 \
                                                            parameters for\
                                                        relu activation!"
                        self.act_params.append(act_params[j])
                else:
                    self.act_params.append([])

        self.build_model()
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    #   set optimizer using self.optimizer_name
    #-------------------------------------------------------------------------
    def set_optimizer(self):
    # determine the optimizer and its parameters
        if (self.optimizer_name == 'SGD'):
            # check opt_params
            if len(self.opt_params) == 0:
                self.opt_params = [0.01,0.0,0.0,False]
            else:
                assert len(self.opt_params) == 4, "Must provide 4 parameters for \
                                              SGD!"
                assert isinstance(self.opt_params[3],bool), "Parameter 4 (nesterov\
                                                        acceleration) must be\
                                                        of type bool!"
            self.optimizer = optimizers.SGD(lr=self.opt_params[0],
                                            momentum=self.opt_params[1],
                                            decay=self.opt_params[2],
                                            nesterov=self.opt_params[3])
        elif (self.optimizer_name == 'RMSprop'):
            # check opt_params
            if len(self.opt_params) == 0:
                self.opt_params = [0.001,0.9,'None',0.0]
            else:
                assert len(self.opt_params) == 4, "Must provide 4 parameters for \
                                              RMSprop!"
            self.optimizer = optimizers.RMSprop(lr=self.opt_params[0],
                                                rho=self.opt_params[1],
                                                epsilon=self.opt_params[2],
                                                decay=self.opt_params[3])
        elif (self.optimizer_name == 'Adagrad'):
            # check opt_params
            if len(self.opt_params) == 0:
                self.opt_params = [0.01,'None',0.0]
            else:
                assert len(self.opt_params) == 3, "Must provide 3 parameters for \
                                              Adagrad!"
            self.optimizer = optimizers.Adagrad(lr=self.opt_params[0],
                                                epsilon=self.opt_params[1],
                                                decay=self.opt_params[2])
        elif (self.optimizer_name == 'Adadelta'):
            # check opt_params
            if len(self.opt_params) == 0:
                self.opt_params = [1.0,0.95,'None',0.0]
            else:
                assert len(self.opt_params) == 4, "Must provide 4 parameters for \
                                              Adadelta!"
            self.optimizer = optimizers.Adadelta(lr=self.opt_params[0],
                                                 rho=self.opt_params[1],
                                                 epsilon=self.opt_params[2],
                                                 decay=self.opt_params[3])
        elif (self.optimizer_name == 'Adam'):
            # check opt_params
            if len(self.opt_params) == 0:
                self.opt_params = [0.001,0.9,0.999,'None',0.0,False]
            else:
                assert len(self.opt_params) == 6, "Must provide 6 parameters for \
                                              Adam!"
                assert isinstance(self.opt_params[5],bool), "Parameter 6 (amsgrad)\
                                                        must be of type bool!"
            self.optimizer = optimizers.Adam(lr=self.opt_params[0],
                                             beta_1=self.opt_params[1],
                                             beta_2=self.opt_params[2],
                                             epsilon=self.opt_params[3],
                                             decay=self.opt_params[4],
                                             amsgrad=self.opt_params[5])
        elif (self.optimizer_name == 'Adamax'):
            # check opt_params
            if len(self.opt_params) == 0:
                self.opt_params = [0.002,0.9,0.999,'None',0.0]
            else:
                assert len(self.opt_params) == 5, "Must provide 5 parameters for \
                                              Adamax!"
            self.optimizer = optimizers.Adamax(lr=self.opt_params[0],
                                               beta_1=self.opt_params[1],
                                               beta_2=self.opt_params[2],
                                               epsilon=self.opt_params[3],
                                               decay=self.opt_params[4])
        elif (self.optimizer_name == 'Nadam'):
            # check opt_params
            if len(self.opt_params) == 0:
                self.opt_params = [0.001,0.9,0.999,1e-7]
            else:
                assert len(self.opt_params) == 4, "Must provide 4 parameters for \
                                              Nadam!"
            self.optimizer = optimizers.Nadam(lr=self.opt_params[0],
                                              beta_1=self.opt_params[1],
                                              beta_2=self.opt_params[2],
                                              epsilon=self.opt_params[3])
    #-------------------------------------------------------------------------


    #-------------------------------------------------------------------------
    #   build and compile sequential model
    #-------------------------------------------------------------------------
    def build_model(self):
        # build the sequential model
        self.model = models.Sequential()
        # now we build and compile the model
        # no biases from the input layer, since the inputs are physical
        self.model.add(layers.Dense(self.topology[1],input_dim=self.topology[0],
                       kernel_initializer=self.initializers[0],
                       use_bias=True))
        self.num_additions = 0
        for i in range( 1, len( self.topology ) - 1 ):
            # This "layer" object applies the activation from the output
            # of the previous
            if(self.activations[i] not in ['elu','relu']):
                self.model.add(layers.Activation(self.activations[i]))
            elif(self.activations[i] == 'elu'):
                self.model.add(layers.ELU(self.act_params[i]))
            elif(self.activations[i] == 'relu'):
                self.model.add(layers.ReLU(self.act_params[i][1],
                                           self.act_params[i][0],
                                           self.act_params[i][2]))
            #   Adding the next layer
            self.model.add(layers.Dense(
                self.topology[i+1],
                kernel_initializer=self.initializers[i],
                use_bias=True)
            )
            self.num_additions += 2
        if(self.activations[-1] not in ['elu','relu']):
            self.model.add(layers.Activation(self.activations[-1]))
        elif(self.activations[-1] == 'elu'):
            self.model.add(layers.ELU(self.act_params[-1]))
        elif(self.activations[-1] == 'relu'):
            self.model.add(layers.ReLU(self.act_params[-1][1],
                                       self.act_params[-1][0],
                                       self.act_params[-1][2]))
        #   We want to retrieve the values from the output
        self.output_function = K.function([self.model.layers[0].input],
                            [self.model.layers[-1].output])
        # now compile the model
        threshold=self.get_threshold()
        if threshold == 0.5 and self.use_auc: # Note that Keras's AUC metric only works for 0-1 range; skip if outside this
            self.model.compile(loss=self.loss,optimizer=self.optimizer,
                               metrics=[tf.keras.metrics.BinaryAccuracy(threshold=threshold),
                                       tf.keras.metrics.AUC(name='auc')])
        else:
            self.model.compile(loss=self.loss,optimizer=self.optimizer,
                               metrics=[tf.keras.metrics.BinaryAccuracy(threshold=threshold)])

    #-------------------------------------------------------------------------
    #   get_layer_weights(self,layer - layer is an index to the desired layer)
    #-------------------------------------------------------------------------
    def get_layer_weights(self,layer):
        if layer >= len(self.topology):
            print("ERROR! index %s exceeds number of layers %s!" % (layer,len(self.topology)))
            return 0
        # check that 'layer' is an actual layer with weights
        try:
            return self.model.layers[layer].get_weights()[0].tolist()
        except:
            print("ERROR! layer %s is not a layer with weights!" % layer)
            return 0
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    #   get_layer_biases(self,layer - layer is an index to the desired layer)
    #-------------------------------------------------------------------------
    def get_layer_biases(self,layer):
        if layer >= len(self.topology):
            print("ERROR! index %s exceeds number of layers %s!" % (layer,len(self.topology)))
            return 0
        # check that 'layer' is an actual layer with weights
        try:
            return self.model.layers[layer].get_weights()[1].tolist()
        except:
            print("ERROR! layer %s is not a layer with weights!" % layer)
            return 0
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    #   get weights
    #-------------------------------------------------------------------------
    def get_weights(self):
        weights = []
        for i in range(len(self.model.layers)):
            try:
                weights.append(self.model.layers[i].get_weights()[0].tolist())
            except:
                continue
        return weights
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    #   set weights
    #-------------------------------------------------------------------------
    def set_weights(self, weights):
        try:
            self.model.set_weights(weights)
        except:
            return
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    #   set weights from file
    #-------------------------------------------------------------------------
    def set_weights_from_file(
        self, 
        weights_file, 
        biases_file
    ):
        weights = []
        biases = []
        with open(weights_file,"r") as file:
            reader = csv.reader(file,delimiter=",")
            topology = next(reader)
            for row in reader:
                weights.append([float(row[i]) for i in range(len(row))])
        with open(biases_file,"r") as file:
            reader = csv.reader(file,delimiter=",")
            next(reader)
            for row in reader:
                biases.append([float(row[i]) for i in range(len(row))])
        topology = [int(topology[i]) for i in range(len(topology))]
        new_weights = []
        index_left = 0
        index_right = topology[0]
        for j in range(len(topology)-1):
            new_weights.append(np.asarray([weights[l] for l in range(index_left,index_right)]))
            if j < len(topology) - 1:
                new_weights.append(np.asarray(biases[j]))
            index_left = index_right
            index_right += topology[j+1]
        try:
            self.model.set_weights(new_weights)
        except:
            return
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    #   get biases
    #-------------------------------------------------------------------------
    def get_biases(self):
        biases = []
        for i in range(len(self.model.layers)):
            try:
                biases.append(self.model.layers[i].get_weights()[1].tolist())
            except:
                continue
        return biases
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    #   save_weights_to_file(self,filename - name to store the weights)
    #-------------------------------------------------------------------------
    def save_weights_to_file(self,filename):
        weights = self.get_weights()
        weights_to_save = [self.topology]
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                weights_to_save.append(weights[i][j])
        with open(filename,"w") as file:
            writer = csv.writer(file,delimiter=",")
            writer.writerows(weights_to_save)
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    #   save_biases_to_file(self,filename - name to store the biases)
    #-------------------------------------------------------------------------
    def save_biases_to_file(self,filename):
        biases = self.get_biases()
        biases_to_save = [self.topology]
        for i in range(len(biases)):
            biases_to_save.append(biases[i])
        with open(filename,"w") as file:
            writer = csv.writer(file,delimiter=",")
            writer.writerows(biases_to_save)
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    #   save_model
    #-------------------------------------------------------------------------
    def save_model(self,filename):
        params = [['#Topology'],self.topology,
                  ['#Optimizer'],[self.optimizer_name],
                  ['#OptParams'],self.opt_params,
                  ['#Activations'],self.activations,
                  ['#ActParams']]
        for j in range(len(self.act_params)):
            if len(self.act_params[j]) == 0:
                params.append(['None'])
            else:
                params.append(self.act_params[j])
        params.append(['#Loss']),
        params.append([self.loss])
        params.append(['#Weights'])
        weights = self.get_weights()
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                params.append(weights[i][j])
        params.append(['#Biases'])
        biases = self.get_biases()
        for i in range(len(biases)):
            params.append(biases[i])
        params.append(['#Normalization'])
        params.append([self.normalization])
        params.append(['#NormalizationParams'])
        for i in range(len(self.normalization_params)):
            params.append(self.normalization_params[i])
        with open(filename,"w") as file:
            writer = csv.writer(file,delimiter=',')
            writer.writerows(params)
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    #   set_model_from_file
    #-------------------------------------------------------------------------
    def set_model_from_file(self,filename):
        params = []
        with open(filename,"r") as file:
            reader = csv.reader(file,delimiter=",")
            for row in reader:
                params.append(row)
        # iterate over each row starting with topology
        self.topology = [int(params[1][i]) for i in range(len(params[1]))]
        self.optimizer_name = params[3][0]
        # set optimizer using optimizer_name
        self.set_optimizer()
        self.opt_params = params[5]
        self.activations = params[7]
        self.act_params = [params[9+i] for i in range(0,len(self.activations))]
        self.loss = params[10+len(self.activations)][0]
        # now for the weights
        weights = []
        biases = []
        weights_start = 12 + len(self.activations)
        num_weights = int(sum([self.topology[i] for i in range(len(self.topology)-1)]))
        for j in range(weights_start,weights_start + num_weights):
            weights.append([float(params[j][l]) for l in range(len(params[j]))])
        biases_start = 13 + len(self.activations) + len(weights)
        for j in range(biases_start,biases_start + len(self.topology)-1):
            biases.append([float(params[j][l]) for l in range(len(params[j]))])
        new_weights = []
        index_left = 0
        index_right = self.topology[0]
        for j in range(len(self.topology)-1):
            new_weights.append(np.asarray([weights[l]
                               for l in range(index_left,index_right)]))
            if j < len(self.topology) - 1:
                new_weights.append(np.asarray(biases[j]))
            index_left = index_right
            index_right += self.topology[j+1]

        # set initializers to default
        if not self.initializers:
            self.initializers = [ki.RandomNormal()
                                 for i in range(len(self.topology)-1)]

        self.build_model()

        self.model.set_weights(new_weights)

        # finally get normalization parameters
        final = 14 + len(self.activations) + len(weights) + len(biases)
        self.normalization = params[final][0]
        if final < len(params)+2:
            for j in range(final+2,len(params)):
                self.normalization_params.append([float(params[j][l])
                                for l in range(len(params[j]))])
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    #   find_normalization_parameters(self,data)
    #-------------------------------------------------------------------------
    def find_normalization_parameters(self,data):
        # determine the normalization parameters
        self.normalization_params = []
        if (self.normalization == 'Standard'):
#           for j in range(len(data[0])):
#               temp_var = [data[i][j] for i in range(len(data))]
#               self.normalization_params.append([np.mean(temp_var),
#                                                 np.std(temp_var)])
            means = np.mean(data,axis=0)
            stds = np.std(data,axis=0)
            for j in range(len(means)):
                self.normalization_params.append([means[j],stds[j]])
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    #   set_norm_params_from_file(self,file)
    #-------------------------------------------------------------------------
    def set_norm_params_from_file(self,input_file):
        with open(input_file, "r") as file:
            reader = csv.reader(file,delimiter=",")
            for row in reader:
                self.normalization_params.append([float(row[1]),float(row[2])])

    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    #   normalize_data(self,data)
    #-------------------------------------------------------------------------
    def normalize_data(self,data):
        # determine the normalization parameters
        if (self.normalization == 'Standard'):
            norm_params = np.array(self.normalization_params)
            return (data-norm_params[:,0])/norm_params[:,1]
        else:
            print("Warning: non-standard normalizations not supported; returning unscaled values.")
            return data
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    #   set_classes(self,answer)
    #                               'tanh' (default)- [-1.0,1.0]
    #                               'elu'           - [-1.0,inf]
    #                               'selu'          - [-1.673,inf]
    #                               'softplus'      - [0.0,inf]
    #                               'softsign'      - [-inf,inf]
    #                               'relu'          - [0.0,inf]
    #                               'sigmoid'       - [0.0,1.0]
    #                               'hard_sigmoid'  - [0.0,1.0]
    #                               'exponential'   - [0.0,inf]
    #                               'linear'        - [-inf,inf]
    #-------------------------------------------------------------------------
    def set_classes(self,answer):
        # change to the correct expected output
        new_answer=np.copy(answer) # ensure output is a numpy array, don't edit original answer
        if (self.activations[-1] == 'tanh' or 
              self.activations[-1] == 'elu' or
              self.activations[-1] == 'softsign' or
              self.activations[-1] == 'linear'):
            return new_answer
        elif (self.activations[-1] == 'selu'):
            low_value = -1.673
            high_value = 1.0
        elif (self.activations[-1] == 'softplus' or
              self.activations[-1] == 'relu' or
              self.activations[-1] == 'exponential' or 
              self.activations[-1] == 'sigmoid' or
              self.activations[-1] == 'hard_sigmoid'):
            low_value = 0.0
            high_value = 1.0
        return low_value*(new_answer!=1.0)+high_value*(new_answer==1.0)
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    #   get_threshold(self)
    #                               'tanh' (default)- [-1.0,1.0]
    #                               'elu'           - [-1.0,inf]
    #                               'selu'          - [-1.673,inf]
    #                               'softplus'      - [0.0,inf]
    #                               'softsign'      - [-inf,inf]
    #                               'relu'          - [0.0,inf]
    #                               'sigmoid'       - [0.0,1.0]
    #                               'hard_sigmoid'  - [0.0,1.0]
    #                               'exponential'   - [0.0,inf]
    #                               'linear'        - [-inf,inf]
    #-------------------------------------------------------------------------
    def get_threshold(self):
        # change according to the expected range of the final activation function
        if (self.activations[-1] == 'tanh'):
            return 0.0
        elif (self.activations[-1] == 'elu'):
            return 0.0 # not obvious where to set this
        elif (self.activations[-1] == 'selu'):
            return 0.0 # not obvious where to set this
        elif (self.activations[-1] == 'softplus' or
              self.activations[-1] == 'relu' or
              self.activations[-1] == 'exponential'):
            return 0.5
        elif (self.activations[-1] == 'softsign' or
              self.activations[-1] == 'linear'):
            return 0.0
        elif (self.activations[-1] == 'sigmoid' or
              self.activations[-1] == 'hard_sigmoid'):
            return 0.5
    #-------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------
    #   train(self, training_data   - should be a list of lists
    #               training_answer - a single list of output values
    #               num_epochs      - number of training epochs
    #               batch           - number of events for batch training
    #               type_norm       - normalization type
    #               sample_weights  - possible weights for samples
    #               normalized      - set to true if training_data is already normalized
    #               shuffle         - if true, randomly partition train/test split
    #               verbose         - sets verbosity for model fitting step
    #-------------------------------------------------------------------------
    def train(
        self,
        training_data,
        training_answer,
        validation_split=0.25,
        num_epochs=1,
        batch=256,
        sample_weights=[],
        normalized=False,
        shuffle=False,
        verbose=False
    ):
        if not normalized:
            train_data = np.copy(training_data)
            # Anytime we are training a network, we must renormalize
            # according to the data
            self.find_normalization_parameters(training_data)
            train_data = self.normalize_data(train_data)
        else:
            train_data = training_data
        # set the training answer to match the expected output
        train_answer = self.set_classes(training_answer)
        # training session
        print("training model...")
        if validation_split > 0.0:
            train_per = 1 - validation_split
            train_num = int(train_per*len(train_data))
            if shuffle:
                ind_order = np.random.permutation(len(train_data))
            else:
                ind_order = np.arange(len(train_data))
            self.test_ints = ind_order[train_num:]
                
            train_x = train_data[ind_order[:train_num]]
            train_y = train_answer[ind_order[:train_num]]
            test_x = train_data[ind_order[train_num:]]
            test_y = train_answer[ind_order[train_num:]]
            if len(sample_weights) == len(train_data):
                sample_weights = np.asarray(sample_weights) # convert from list to array if needed
                train_w = sample_weights[ind_order[:train_num]]
                test_w = sample_weights[ind_order[train_num:]]
                self.history.append(self.model.fit(np.asarray(train_x), np.asarray(train_y),
                        validation_data=(np.asarray(test_x), np.asarray(test_y), np.asarray(test_w)),
                        epochs=num_epochs, batch_size=batch,
                        sample_weight=np.asarray(train_w),verbose=verbose, callbacks=[self.time_callback]))
            else:
                self.history.append(self.model.fit(np.asarray(train_x), np.asarray(train_y),
                       validation_data=(np.asarray(test_x),np.asarray(test_y)),
                       epochs=num_epochs, batch_size=batch,verbose=verbose, callbacks=[self.time_callback]))
        else:
            self.history.append(self.model.fit(np.asarray(train_data),np.asarray(train_answer),
                        epochs=num_epochs, batch_size=batch,
                        sample_weight=np.asarray(sample_weights),verbose=verbose, callbacks=[self.time_callback]))
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    #   train_subsets(self, training_data   - should be a list of lists
    #               training_answer - a single list of output values
    #               num_epochs      - number of training epochs
    #               batch           - number of events for batch training
    #               type_norm       - normalization type
    #               sample_weights  - possible weights for samples
    #               normalized      - set to true if training_data is already normalized
    #               shuffle         - if true, randomly partition train/test split
    #               verbose         - sets verbosity for model fitting step
    #-------------------------------------------------------------------------
    def train_subsets(
        self,
        training_data,
        training_answer,
        validation_split=0.25,
        num_epochs=1,
        batch=256,
        training_weights=[],
        num_subsets=10,
        signal_subsets=-1,
        background_subsets=-1,
        shuffle=False,
        verbose=False
    ):
        # set the training answer to match the expected output
        train_answer = self.set_classes(training_answer)
        # training session
        print("training model...")
        signal = [i for i in range(len(training_data)) if training_answer[i][0] == 1.0]
        background = [i for i in range(len(training_data)) if training_answer[i][0] == 0.0]
        if signal_subsets!= -1:
            num_signal_subset = int(len(signal)/num_subsets)
        else:
            num_signal_subset = 1
        if background_subsets!= -1:
            num_background_subset = int(len(background)/num_subsets)
        else:
            num_background_subset = 1
        for ii in range(len(num_subsets)):
            if signal_subsets != -1:
                sig_inds = signal[ii*num_signal_subset:(ii+1)*num_signal_subset]
            else:
                sig_inds = signal
            if background_subsets != -1:
                back_inds = background[ii*num_background_subset:(ii+1)*num_background_subset]
            inds = np.concatenate((sig_inds,back_inds))
            train_data = training_data[inds]
            train_answer = training_answer[inds]
            sample_weights = training_weights[inds]
            if validation_split > 0.0:
                train_per = 1 - validation_split
                train_num = int(train_per*len(train_data))
                if shuffle:
                    ind_order = np.random.permutation(len(train_data))
                else:
                    ind_order = np.arange(len(train_data))
                self.test_ints = ind_order[train_num:]
                    
                train_x = train_data[ind_order[:train_num]]
                train_y = train_answer[ind_order[:train_num]]
                test_x = train_data[ind_order[train_num:]]
                test_y = train_answer[ind_order[train_num:]]
                if len(sample_weights) == len(train_data):
                    sample_weights = np.asarray(sample_weights) # convert from list to array if needed
                    train_w = sample_weights[ind_order[:train_num]]
                    test_w = sample_weights[ind_order[train_num:]]
                    self.history.append(self.model.fit(np.asarray(train_x), np.asarray(train_y),
                            validation_data=(np.asarray(test_x), np.asarray(test_y), np.asarray(test_w)),
                            epochs=num_epochs, batch_size=batch,
                            sample_weight=np.asarray(train_w),verbose=verbose, callbacks=[self.time_callback]))
                else:
                    self.history.append(self.model.fit(np.asarray(train_x), np.asarray(train_y),
                        validation_data=(np.asarray(test_x),np.asarray(test_y)),
                        epochs=num_epochs, batch_size=batch,verbose=verbose, callbacks=[self.time_callback]))
            else:
                self.history.append(self.model.fit(np.asarray(train_data),np.asarray(train_answer),
                            epochs=num_epochs, batch_size=batch,
                            sample_weight=np.asarray(sample_weights),verbose=verbose, callbacks=[self.time_callback]))
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    #   train__file_subsets(self, training_data   - should be a list of lists
    #               training_answer - a single list of output values
    #               num_epochs      - number of training epochs
    #               batch           - number of events for batch training
    #               type_norm       - normalization type
    #               sample_weights  - possible weights for samples
    #               normalized      - set to true if training_data is already normalized
    #               shuffle         - if true, randomly partition train/test split
    #               verbose         - sets verbosity for model fitting step
    #-------------------------------------------------------------------------
    def train_file_subsets(
        self,
        data_folder,
        var_set=[],
        validation_split=0.25,
        num_epochs=10,
        batch=256,
        weight=-1,
        num_subsets=10,
        shuffle=False,
        verbose=False,
        metrics=False
    ):
        mis, aucs, leakages = [], [], []
        for ii in range(num_subsets):
            train_file = h5py.File(data_folder+'training%s.h5' % ii, 'r')
            train_data = np.array(train_file.get('training data'))
            train_answer = np.array(train_file.get('training answer'))
            if weight != -1:
                sample_weights = np.array(train_file.get('weight'))
            else:
                sample_weights = []
            if validation_split > 0.0:
                if len(sample_weights) == len(train_data):
                    self.history.append(self.model.fit(train_data, train_answer,
                            validation_split=validation_split, validation_batch_size=batch,
                            epochs=num_epochs, batch_size=batch,
                            sample_weight=sample_weights, verbose=verbose, 
                            callbacks=[self.time_callback]))
                else:
                    self.history.append(self.model.fit(train_data, train_answer,
                            validation_split=validation_split, validation_batch_size=batch,
                            epochs=num_epochs, batch_size=batch,
                            verbose=verbose, 
                            callbacks=[self.time_callback]))
                if metrics:
                    results = self.evaluate(train_data,train_answer)
                    temp_s = [[results[i][0]] for i in range(len(results)) if results[i][1] == 1]
                    temp_b = [[results[i][0]] for i in range(len(results)) if results[i][1] != 1]
                    if len(sample_weights) == len(train_data):
                        w_s = [sample_weights[i] for i in range(len(sample_weights)) if results[i][1] == 1]
                        w_b = [sample_weights[i] for i in range(len(sample_weights)) if results[i][1] != 1]
                        mis.append(mi.mi_binary_weights(temp_s,temp_b,w_s,w_b,k=1))
                    else:
                        mis.append(mi.mi_binary(temp_s,temp_b,k=1)) 
                    fpr, tpr, thres, auc = self.plot_weighted_auc(train_data,train_answer,sample_weights,show=False,normalized=True)
                    leakages.append(fpr[bisect.bisect_left(tpr,0.5)])
                    aucs.append(auc)

            else:
                if len(sample_weights) == len(train_data):
                    self.history.append(self.model.fit(train_data, train_answer,
                            epochs=num_epochs, batch_size=batch,
                            sample_weight=sample_weights, verbose=verbose, 
                            callbacks=[self.time_callback]))
                else:
                    self.history.append(self.model.fit(train_data, train_answer,
                            epochs=num_epochs, batch_size=batch,
                            verbose=verbose, 
                            callbacks=[self.time_callback]))
        return mis,aucs,leakages
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    #   evaluate(self, testing_data   - should be a list of lists
    #                  testing_answer - a single list of output values
    #                  score_output   - whether to display the score
    #                  batch          - batch amount for the score
    #-------------------------------------------------------------------------
    def evaluate(
        self,
        testing_data,
        testing_answer,
        score_output=True,
        batch=256,
        normalize=False,
    ):
        test_data = np.copy(testing_data)
        #   We don't want to normalize the actual testing data, only a copy of it
        if normalize:
            test_data = self.normalize_data(test_data)
        # set the testing answer to match the expected output
        test_answer = self.set_classes(testing_answer)
        if (score_output == True):
            score = self.model.evaluate(np.array(test_data),
                                        np.array(test_answer),
                                        batch_size=batch,
                                        verbose=0)
            #   Prints a score for the network based on the training data
            print('Score: %s' % (score))
        activations = self.output_function(np.asarray(test_data))
        return [[activations[0][i][0], test_answer[i]] for i in range(len(testing_answer))]
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    #   save_results_to_file(self, filename)
    #-------------------------------------------------------------------------
    def save_results_to_file(
        self,
        filename,
        data,
        results
    ):
        events = np.concatenate((data,results),axis=1)
        with open(filename,"w") as file:
            writer = csv.writer(file,delimiter=",")
            writer.writerows(events)
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    #   plot_history(self)
    #-------------------------------------------------------------------------
    def plot_history(
        self, 
        show=True, 
        save=True, 
        filename='History'
    ):
        bin_accuracy = []
        val_accuracy = []
        loss = []
        val_loss = []
        for l in range(len(self.history)):
            bin_accuracy += self.history[l].history['binary_accuracy']
            val_accuracy += self.history[l].history['val_binary_accuracy']
            loss += self.history[l].history['loss']
            val_loss += self.history[l].history['val_loss']
        fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(11,5),
                                constrained_layout=True)
        ax = axs[0]
        ax.plot(bin_accuracy)
        ax.plot(val_accuracy)
        ax.set_title('Model accuracy')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend(['Train', 'Validation'], loc='upper left')

        # Plot training & validation loss values
        ax = axs[1]
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title('Model loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend(['Train', 'Validation'], loc='upper left')

        fig.suptitle('Training/Validation Error and Accuracy vs. Epoch')
        if(save):
            plt.savefig(filename + '.png')
#           with open(filename+".csv","w") as file:
#               writer = csv.writer(file,delimiter=",")
#               writer.writerows([self.history.history['binary_accuracy'],
#                                 self.history.history['val_binary_accuracy'],
#                                 self.history.history['loss'],
#                                 self.history.history['val_loss']])
        if(show):
            plt.show()
        else:
            plt.close()
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    #   plot_auc(self, results)
    #   note that answer can be w/ or w/o scaling in set_classes()
    #-------------------------------------------------------------------------
    def plot_auc(
        self, 
        data,  
        answer,
        show=True, 
        save=True, 
        filename='AccRej', 
        normalized=False
    ):
        
        if not normalized:
            test_data = np.copy(data)
            #   We don't want to normalize the actual testing data, only a copy of it
            test_data = np.asarray(self.normalize_data(test_data))
        else:
            test_data = data
        # set the testing answer to match the expected output
        y_pred = self.model.predict(test_data).ravel()
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(answer, y_pred)
        auc_keras = auc(fpr_keras, tpr_keras)
        #with open(filename+".csv","w") as file:
        #   writer = csv.writer(file,delimiter=",")
        #   writer.writerows([fpr_keras,tpr_keras,thresholds_keras])
        if (save or show):
            fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(11,5),
                                    constrained_layout=True)
            ax = axs[0]
            ax.plot([0, 1], [0, 1], 'k--')
            ax.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
            ax.set_xlabel('False positive rate')
            ax.set_ylabel('True positive rate')
            ax.set_title('ROC curve')
            ax.legend(loc='best')
            # Log scale for false positives (to focus on low leakage)
            ax = axs[1]
            plt.xlim(1e-7, 1.0)
            plt.ylim(0, 1)
            ax.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
            ax.set_xscale('log')
            ax.set_xlabel('False positive rate')
            ax.set_ylabel('True positive rate')
            ax.set_title('ROC curve (log leakage)')
            ax.legend(loc='best')
            fig.suptitle('True positive vs. False positive rate')
        if(save):
            plt.savefig(filename + '.png')
        if(show):
            plt.show()
        else:
            plt.close()
        return fpr_keras, tpr_keras, thresholds_keras, auc_keras
    #-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    #   plot_weighted_auc(self, results)
    #   note that answer can be w/ or w/o scaling in set_classes()
    #-------------------------------------------------------------------------
    def plot_weighted_auc(
        self, 
        data, 
        answer, 
        weights=None,
        show=True, 
        save=True, 
        filename='AccRej', 
        normalized=False
    ):
        if not normalized:
            test_data = np.copy(data)
            #   We don't want to normalize the actual testing data, only a copy of it
            test_data = np.asarray(self.normalize_data(test_data))
        else:
            test_data = data
        # set the testing answer to match the expected output
        y_pred = self.model.predict(test_data).ravel()
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(answer, y_pred, sample_weight=weights)
        auc_keras = auc(fpr_keras, tpr_keras)
        with open(filename+".csv","w") as file:
            writer = csv.writer(file,delimiter=",")
            writer.writerows([fpr_keras,tpr_keras,thresholds_keras])
        fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(11,5),
                                constrained_layout=True)
        ax = axs[0]
        ax.plot([0, 1], [0, 1], 'k--')
        ax.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.set_title('ROC curve')
        ax.legend(loc='best')
        # Log scale for false positives (to focus on low leakage)
        ax = axs[1]
        plt.xlim(1e-7, 1.0)
        plt.ylim(0, 1)
        ax.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
        ax.set_xscale('log')
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.set_title('ROC curve (log leakage)')
        ax.legend(loc='best')
        fig.suptitle('True positive vs. False positive rate')
        if(save):
            plt.savefig(filename + '.png')
        if(show):
            plt.show()
        else:
            plt.close()
        return fpr_keras, tpr_keras, thresholds_keras, auc_keras

#-----------------------------------------------------------------------------