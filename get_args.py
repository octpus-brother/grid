import preprocess as pp
import HTSAD
import deeplog
import PCArecon
import AE
import LSTM


def get_args(modelname):
    args = {"class_num": pp.class_num, "categorical_num": pp.categorical_num, "numerical_num": pp.numerical_num}
    if modelname == "HTSAD":
        model = HTSAD
        args["model"] = "HTSAD"
        args["WindowSize"] = 300
        args["Stride"] = 1
        args["Epoches"] = 10
        args["Embedding_sz"] = 8
        args["Output_sz"] = 16
        args["Eventfilter_sz"] = 16
        args["BatchSize"] = 1
        args["WeightDecay"] = 0.001
        args["LearningRate"] = 0.0005
        args["Dim"] = 4
        args["LossThreshold"] = 0.1
        args["TrainThreshold"] = 0.01
    elif modelname == "deeplog":
        model = deeplog
        args["model"] = "deeplog"
        args["WindowSize"] = 300
        args["input_size"] = 1
        args["hidden_size"] = 64
        args["num_layers"] = 2
        args["BatchSize"] = 16
        args["LearningRate"] = 0.0005
        args["Epoches"] = 10
    elif modelname == "LSTM":
        model = LSTM
        args["model"] = "LSTM"
        args["WindowSize"] = 300
        args["Stride"] = 1
        args["Epoches"] = 10
        args["Embedding_sz"] = 8
        args["Output_sz"] = 16
        args["BatchSize"] = 4
        args["WeightDecay"] = 0.001
        args["LearningRate"] = 0.0005
        args["Dim"] = 4
        args["LossThreshold"] = 0.1
        args["TrainThreshold"] = 0.01
    elif modelname == "PCArecon":
        model = PCArecon
        args["model"] = "PCArecon"
        args["explained_var"] = 0.9
    elif modelname == "AE":
        model = AE
        args["model"] = "AE"
        args["WindowSize"] = 300
        args["Stride"] = 1
        args["Epoches"] = 10
        args["Embedding_sz"] = 4
        args["Hidden_sz"] = 8
        args["Eventfilter_sz"] = 16
        args["BatchSize"] = 16
        args["WeightDecay"] = 0.001
        args["LearningRate"] = 0.0005
    else:
        model = None
        print("No Such Model!")
    return model, args


def get_test_args(modelname):
    test_args = {}
    if modelname == "HTSAD":
        test_args["TestThreshold"] = 1.05
    elif modelname == "deeplog":
        test_args["candidates_num"] = 4
    elif modelname == "LSTM":
        test_args["TestThreshold"] = 1.05
    elif modelname == "PCArecon":
        test_args["threshold"] = 2.0
    elif modelname == "AE":
        test_args["threshold"] = 0.03
    else:
        print("No Such Model!")
    return test_args
