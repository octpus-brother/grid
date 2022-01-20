import get_args
import datetime

modellist = ["HTSAD", "LSTM", "deeplog", "AE"]


def trainer(train_data, modelname, modelnum, severname, device):

    model, args = get_args.get_args(modelname)
    args["modelnum"] = modelnum
    args["severname"] = severname

    model.train(args, device, train_data)


def tester(test_data, modelname, modelnum, severname, device):
    model, _ = get_args.get_args(modelname)
    test_args = get_args.get_test_args(modelname)
    test_args["modelnum"] = modelnum
    test_args["severname"] = severname
    test_args["BatchSize"] = 256

    model.test(test_args, device, test_data)


def getbaseline(sever, ans_path):
    f = open(ans_path, "w")
    f.write("baseline:\n")
    baseline = []
    for item in sever.record:
        baseline.append(item.level)
    f.write(str(baseline))
    f.write("\n")


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")