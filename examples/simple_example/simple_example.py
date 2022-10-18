"""
    This example uses data generated from NEST, which can be found here:
    https://github.com/swkravitz/dance-ml-2020-tutorial.

    Author:
        Scott Kravitz [swkravitz@lbl.gov]
"""
from lux_ml.mlp import MLP
from lux_ml.utils import generate_binary_training_testing_data, convert_root_to_npz
from lux_ml.subsets_mi import find_ML_transform, apply_ML_transform


if __name__ == "__main__":

    # create npz file of root data
    convert_root_to_npz(
        '../../data/CH3TData',
        'nest',
        ['x_cm','y_cm','s1Area_phd','s2Area_phd']
    )
    convert_root_to_npz(
        '../../data/DDData',
        'nest',
        ['x_cm','y_cm','s1Area_phd','s2Area_phd']
    )

    # generate testing and training data
    data = generate_binary_training_testing_data(
        '../../data/DDData.npz',
        '../../data/CH3TData.npz',
        labels=['arr_0','arr_0'],
        symmetric_signals=True,
        testing_ratio=0.0,
        var_set=[0,1,2,3]
    )

    # setup mlp
    mlp = MLP(
        [4,10,3,1],
        optimizer='Nadam',
        activation=['hard_sigmoid','tanh','sigmoid','sigmoid'],
        initializer=['normal','normal','normal'],
        init_params=[[0,1,0],[0,1,1],[0,1,5]]
    )
    mlp.save_weights_to_file('weights_before.csv')
    mlp.save_biases_to_file('biases_before.csv')
    mlp.save_model('model_before.csv')

    # train mlp
    mlp.train(
        data[0],
        data[1],
        num_epochs=20,
        batch=25,
        verbose=True
    )
    mlp.save_weights_to_file('weights_after.csv')
    mlp.save_biases_to_file('biases_after.csv')
    mlp.save_model('model_after.csv')

    # test mlp
    mlp.plot_history()
    results = mlp.evaluate(data[0],data[1])
    mlp.plot_auc(data[0],data[1])

    # save test results
    test_results = mlp.evaluate(data[0][:10],data[1][:10])
    mlp.save_results_to_file(
        'results.csv',
        data[0][:10],
        test_results
    )

    # make a uniform ML transformation
    # from the model
    find_uniform_transform(
        exp_folder = "uniform_output/",
        signal = '../../data/CH3TData.npz',
        background = '../../data/DDData.npz',
        background_files=[],
        filetype='npz',
        treename='nest',
        input_names=['x_cm','y_cm','s1Area_phd','s2Area_phd'],
        model_file='model_after.csv',
        weighted=False,
        mass=0,

    )
    apply_ML_transform(
        exp_folder = "uniform_output/",
        filename = '../../data/CH3TData.npz',
        filetype='npz',
        treename='nest',
        input_names=['x_cm','y_cm','s1Area_phd','s2Area_phd'],
        output_name='ML_output',
        output_file='uniform.root',
        model_file='model_after.csv',
        weighted=False,
        make_output_uniform=True,
        interp_fname_label="uniform_output/Uniform_interp.txt"
    )
