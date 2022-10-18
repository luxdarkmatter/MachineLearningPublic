# Machine Learning Public

This repository contains public code for the paper *["Fast and Flexible Analysis of Direct Dark Matter Search Data with Machine Learning"](https://arxiv.org/abs/2201.05734)* by the LUX Collaboration.
    - ArXiv: <https://arxiv.org/abs/2201.05734>
    - Published in Physical Review D, ... <>

## Installation and basic usage

To use the code in this repository, simply install it using pip.
```bash
git clone https://github.com/luxdarkmatter/MachineLearningPublic
cd MachineLearningPublic
pip install -r requirements.txt .
```

This will make the package *lux_ml* available in python, which has a few subpackages:
```python
import lux_ml.mi as mi
import lux_ml.mlp as mlp
import lux_ml.subsets_mi as subsets_mi
from lux_ml.utils import *
```

A simple example script is located in the folder */examples/simple_example/* which uses the DD and CH3T data in the */data/* folder.  The DD and CH3T data was generated using **[NEST](https://nest.physics.ucdavis.edu/)** as part of a *[ML workshop]*(https://github.com/swkravitz/dance-ml-2020-tutorial).

The tensorflow version used in this repository is *tensorflow=2.10.0*.

## Example

To use the MLP for training a model, data must first be prepared in the appropriate .npz format.  This can be accomplished with a set of functions (as shown in the *simple_example*):
```python
from lux_ml.utils import convert_root_to_npz
# create npz file of root data
# first convert the tritrium data (background)
convert_root_to_npz(
    '../../data/CH3TData',                      # location and name of the file (no .root extension)
    'nest',                                     # name of the TTree holding the data
    ['x_cm','y_cm','s1Area_phd','s2Area_phd']   # names of the variables we want to extract from the data
)
# then convert the DD data (signal)
convert_root_to_npz(
    '../../data/DDData',
    'nest',
    ['x_cm','y_cm','s1Area_phd','s2Area_phd']
)
```
Once the data has been properly converted, we can load it in using another function from *lux_ml.utils*:
```python
from lux_ml.utils import generate_binary_training_testing_data
# generate testing and training data
data = generate_binary_training_testing_data(
    '../../data/DDData.npz',    # signal file
    '../../data/CH3TData.npz',  # background file
    labels=['arr_0','arr_0'],   # default names of the arrays saved in .npz format
    symmetric_signals=True,     # whether to use equal amounts of signal/background
    testing_ratio=0.0,          # how much of the data to set aside for testing
    var_set=[0,1,2,3]           # which variables we want to use
)
```

Next we'll want to create our MLP object, which we can do as follows:
```python
from lux_ml.mlp import MLP
# setup mlp
mlp = MLP(
    [4,10,3,1],                                                 # topology of the network
    optimizer='Nadam',                                          # optimizer to use
    activation=['hard_sigmoid','tanh','sigmoid','softsign'],    # activation functions for each layer
    initializer=['normal','normal','normal'],                   # initializer for weights in each layer
    init_params=[[0,1,0],[0,1,1],[0,1,5]]                       # initialization parameters for the weights
)
mlp.save_weights_to_file('weights_before.csv')      # save the initial weights values
mlp.save_biases_to_file('biases_before.csv')        # save the initial bias weight values
mlp.save_model('model_before.csv')                  # save the model dictionary before training
```

To train, we simply run the train method, feeding in the data we generated from before:
```python
# train mlp
mlp.train(
    data[0],            # input data
    data[1],            # input targets
    num_epochs=100,     # number of epochs to train for
    batch=25,           # batch size for training
    verbose=True        # whether to print the output of training to console
)
mlp.save_weights_to_file('weights_after.csv')   # save the final weight values
mlp.save_biases_to_file('biases_after.csv')     # save the final bias weight values
mlp.save_model('model_after.csv')               # save the trained model dictionary
```

We can then evaluate the model and print some statistics from training.
```python
 # test mlp
mlp.plot_history()                          # plot loss/accuracy curves
results = mlp.evaluate(data[0],data[1])     # evaluate the network on the original input data
mlp.plot_auc(data[0],data[1])               # plot AUC curves for the trained model

# save test results
test_results = mlp.evaluate(data[0][:10],data[1][:10])  # grab some results
mlp.save_results_to_file(   # save the results to file.
    'results.csv',
    data[0][:10],
    test_results
)
```

## Uniform ML Transformation
To ease the PLR algorithm, we apply a uniform transformation to the neural network output so that the distribution over the output variable is uniform.  This makes it easier for the PLR to vary bin contents, since each bin has the same integral area.  To use this functionality, you need to pass both the signal file and background file, as well as the trained network parameters to the "find_uniform_transform" function, which has the following arguments:
```python
def find_uniform_transform(
    exp_folder: str,                 # folder to store the results
    signal: str,                     # input file for signal (either root or npz)
    background: str,                 # input file for background (either root or npz)
    background_files: List[str],     # input files for separate background components
    filetype: str='ROOT',            # filetype (either 'ROOT' or 'npz')
    treename: str='summary',         # name of the TTree if ROOT format
    input_names: List[str]=[''],     # names of the variables to be used
    swap_order: List[int]=[],        # order to put the variables in if network takes different order
    weighted: bool='False',          # whether their is a weight variable in the ROOT file
    model_file: str='',              # location of the model file
    mass: str=''                     # mass of the signal
):
```
After model training is done, the trained and saved model can be applied to any input data, provided that the data is normalized correctly for the network. The uniform transformation can also be optionally included in the output, if the uniform transformation parameters were found using the function above. This can be done with the "apply_ML_transform" function, which has the following arguments:
```python
def apply_ML_transform(
    exp_folder: str,                 # folder to store the results
    filename: str,                   # input file (either root or npz)
    filetype: str='ROOT',            # filetype (either 'ROOT' or 'npz')
    treename: str='summary',         # name of the TTree if ROOT format
    input_names: List[str]=[''],     # names of the variables to be used
    output_name: str='',             # name of the output variable to be used
    swap_order: List[int]=[],        # order to put the variables in if network takes different order
    weighted: bool='False',          # whether their is a weight variable in the ROOT file
    model_file: str='',              # location of the model file
    output_file: str='',             # where you want the output to go,
    make_output_uniform: bool=False, # whether to turn the output info uniform from 1/2 sig and 1/2 bkg
    interp_fname_label: str=''       # file name for uniform transformation
) -> None:
```
An example usage is shown in the simple_example script.

## Transfer Learning

The script *lux_ml.subsets_mi* contains a complex wrapper for creating and running a MLP model on data which it splits into subsets and calculates mutual information to use as a benchmark for training.  The method which does this has the following arguments:
```python
def generate_model(
    exp_folder: str,            # location to store results
    data_folder: str,           # location of the data
    process_folder: str,        # location to store processed data
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
)
```

The notebook */notebooks/TransferLearning.ipynb* shows a basic usage (assuming one has a set of WIMP signal files and a background file) using parallel processing to evaluate several iterations of the model at once.  Refer to the [paper](https://arxiv.org/abs/2201.05734) for more details on how transfer learning was used.
