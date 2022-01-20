import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import pickle
import csv
import getdata as gd


def get_dataset(args, trainlogs):
    seq = gd.log_processor(trainlogs)
    s_seq = []
    for item in seq:
        s_seq.append(item[1][0])
    s_seq = torch.tensor(s_seq).reshape((1, -1, 1))
    inputs = []
    outputs = []
    for i in range(s_seq.shape[1] - args["WindowSize"]):
        inputs.append(s_seq[0][i:i + args["WindowSize"]])
        outputs.append(s_seq[0][i + args["WindowSize"]])
    inputs = torch.cat(inputs, 1).reshape(-1, args["WindowSize"])
    outputs = torch.cat(outputs).reshape(-1, 1)
    dataset = TensorDataset(inputs, outputs)
    return dataset


class Deeplog_key(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Deeplog_key, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train(args, device, trainlogs):
    layer = Deeplog_key(args["input_size"], args["hidden_size"], args["num_layers"], args["class_num"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(layer.parameters(), lr=args["LearningRate"])
    train_data = get_dataset(args, trainlogs)
    train_loader = DataLoader(train_data, batch_size=args["BatchSize"], shuffle=True)

    layer.train()
    loss_record = []
    for epoch in range(args["Epoches"]):

        loss_record.append("epoch:" + str(epoch))
        train_loss = 0
        for step, (seq, label) in enumerate(train_loader):
            seq = seq.clone().detach().view(-1, args["WindowSize"], args["input_size"]).to(device)
            seq = seq.double()
            output = layer(seq)
            label = label.squeeze()
            loss = criterion(output, label.to(device))
            # print("{:.4f}".format(loss.data))
            loss_record.append(str(loss.data))

            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

    modelnum = args["modelnum"]
    severname = args["severname"]
    if not os.path.exists('./models/deeplog/epoch{}/{}/'.format(modelnum, severname)):
        os.makedirs('./models/deeplog/epoch{}/{}/'.format(modelnum, severname))
    torch.save(layer.state_dict(), './models/deeplog/epoch{}/{}/deeplogmodel.pth'.format(modelnum, severname))
    torch.save(optimizer.state_dict(), './models/deeplog/epoch{}/{}/deeplogoptim.pth'.format(modelnum, severname))
    f = open('./models/deeplog/epoch{}/{}/deeplogargs.pkl'.format(modelnum, severname), 'wb')
    pickle.dump(args, f)
    f.close()
    f = open("./models/deeplog/epoch{}/{}/deeplogtrain.csv".format(modelnum, severname), "w")
    writer = csv.writer(f)
    for i in args:
        writer.writerow((i, args[i]))
    writer.writerow("")
    for i in loss_record:
        writer.writerow((str(i),))
    f.close()


def test(test_args, device, testlogs):
    modelnum = test_args["modelnum"]
    severname = test_args["severname"]
    candidates_num = test_args["candidates_num"]
    f = open('./models/deeplog/epoch{}/{}/deeplogargs.pkl'.format(modelnum, severname), 'rb')
    args = pickle.load(f)
    f.close()

    layer = Deeplog_key(args["input_size"], args["hidden_size"], args["num_layers"], args["class_num"]).to(device)
    layer.load_state_dict(torch.load('./models/deeplog/epoch{}/{}/deeplogmodel.pth'.format(modelnum, severname)))
    layer.to(device)

    test_data = get_dataset(args, testlogs)
    test_loader = DataLoader(test_data, batch_size=test_args["BatchSize"], shuffle=False)

    layer.eval()
    sth_wrong = []
    if not os.path.exists('./return/epoch{}/{}/'.format(modelnum, severname)):
        os.makedirs('./return/epoch{}/{}/'.format(modelnum, severname))
    f = open("./return/epoch{}/{}/deeplogtest.csv".format(modelnum, severname), "w")
    writer = csv.writer(f)

    for seq, label in test_loader:
        seq = seq.clone().detach().view(-1, args["WindowSize"], args["input_size"]).to(device)
        seq = seq.double()
        output = layer(seq)
        label = label.squeeze()
        predicted = output.argsort(dim=1)[:, -candidates_num:]
        # print(step, label, predicted[-candidates_num:])
        for i in range(label.shape[0]):
            writer.writerow((str(label[i]), str(predicted[i])))
            if label[i] not in predicted[i]:
                sth_wrong.append(1)
            else:
                sth_wrong.append(0)

    f.close()

    print("        {} anomalies detected in {} seqs.".format(sum(sth_wrong), len(sth_wrong)))
    f = open("./return/epoch{}/{}/ans.txt".format(modelnum, severname), "a")
    f.write("deeplog:\n" + str(sth_wrong) + "\n")
    f.close()
