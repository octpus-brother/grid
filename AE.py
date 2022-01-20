import torch
import torch.nn as nn
from torch import optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import preprocess as pp
import getdata as gd
from getdata import SeqDataset
import os
import pickle
import csv
import matplotlib.pyplot as plt
import time


class AE(nn.Module):
    def __init__(self, window_size, hidden_size, embedding_size, device):
        super().__init__()
        self.seq_length = window_size * embedding_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.device = device

        self.linear_class = nn.Linear(pp.class_num, embedding_size, bias=False)
        self.linear_cat = nn.Linear(pp.categorical_num, embedding_size, bias=False)
        self.linear_num = nn.Linear(pp.numerical_num, embedding_size, bias=False)

        dec_steps = 3 ** np.arange(max(np.ceil(np.log2(self.hidden_size)), 2), np.log2(self.seq_length))[1:-2]
        dec_setup = np.concatenate([[self.hidden_size], dec_steps.repeat(2), [self.seq_length]])
        enc_setup = dec_setup[::-1]
        layers = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()] for a, b in enc_setup.reshape(-1, 2)]).flatten()[:-1]
        self._encoder = nn.Sequential(*layers)
        layers = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()] for a, b in dec_setup.reshape(-1, 2)]).flatten()[:-1]
        self._decoder = nn.Sequential(*layers)

    def forward(self, event, vc, vn):
        x = self.linear_class(event) + 2.0 * (self.linear_cat(vc) + torch.tanh(self.linear_num(vn)))
        x = x.reshape(-1, x.shape[2], x.shape[1])
        flattened_x = x.view(x.shape[0], -1)
        enc = self._encoder(flattened_x)
        dec = self._decoder(enc)
        reconstructed_x = dec.view(x.shape)
        return reconstructed_x, x


def train(args, device, trainlogs):
    layer = AE(args["WindowSize"], args["Hidden_sz"], args["Embedding_sz"], device).to(device)
    optimizer = optim.Adam(layer.parameters(), lr=args["LearningRate"], weight_decay=args["WeightDecay"])

    s_seq, time_seq, vc_seq, vn_seq = gd.log2tensors(trainlogs, device)
    data_s = gd.split_seq(s_seq, args["WindowSize"], args["Stride"])
    data_t = gd.split_seq(time_seq, args["WindowSize"], args["Stride"])
    data_vc = gd.split_seq(vc_seq, args["WindowSize"], args["Stride"])
    data_vn = gd.split_seq(vn_seq, args["WindowSize"], args["Stride"])

    train_data = SeqDataset(data_s, data_t, data_vc, data_vn)
    train_loader = DataLoader(train_data, batch_size=args["BatchSize"], shuffle=True)

    layer.train()
    loss_record = []
    for epoch in range(args["Epoches"]):
        loss_record.append("epoch:" + str(epoch))
        count = 0
        for s, t, vc, vn in train_loader:
            count += 1
            optimizer.zero_grad()
            x_out, x = layer(s, vc, vn)
            loss = nn.MSELoss(reduction="mean")(x_out, x)
            if count % 20 == 1:
                progress = epoch / args["Epoches"] + count / (len(train_loader) * args["Epoches"])
                print("        {:.4f}\t{:.2%}".format(loss.data, progress))
            # else:
            #     print("{:.4f}".format(loss.data))
            loss_record.append(str(loss.data))
            loss.backward()
            optimizer.step()

    modelnum = args["modelnum"]
    severname = args["severname"]
    if not os.path.exists('./models/AE/epoch{}/{}/'.format(modelnum, severname)):
        os.makedirs('./models/AE/epoch{}/{}/'.format(modelnum, severname))
    torch.save(layer.state_dict(), './models/AE/epoch{}/{}/AEmodel.pth'.format(modelnum, severname))
    torch.save(optimizer.state_dict(), './models/AE/epoch{}/{}/AEoptim.pth'.format(modelnum, severname))
    f = open('./models/AE/epoch{}/{}/AEargs.pkl'.format(modelnum, severname), 'wb')
    pickle.dump(args, f)
    f.close()
    f = open("./models/AE/epoch{}/{}/AEtrain.csv".format(modelnum, severname), "w")
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
    f = open('./models/AE/epoch{}/{}/AEargs.pkl'.format(modelnum, severname), 'rb')
    args = pickle.load(f)
    f.close()

    layer = AE(args["WindowSize"], args["Hidden_sz"], args["Embedding_sz"], device)

    layer.load_state_dict(torch.load('./models/AE/epoch{}/{}/AEmodel.pth'.format(modelnum, severname)))

    layer.to(device)

    tests_seq, testtime_seq, testvc_seq, testvn_seq = gd.log2tensors(testlogs, device)

    testdata_s = gd.split_seq(tests_seq, args["WindowSize"], args["Stride"])
    testdata_t = gd.split_seq(testtime_seq, args["WindowSize"], args["Stride"])
    testdata_vc = gd.split_seq(testvc_seq, args["WindowSize"], args["Stride"])
    testdata_vn = gd.split_seq(testvn_seq, args["WindowSize"], args["Stride"])

    test_data = SeqDataset(testdata_s, testdata_t, testdata_vc, testdata_vn)
    test_loader = DataLoader(test_data, batch_size=test_args["BatchSize"], shuffle=False)

    layer.eval()
    loss_seq = []
    sth_wrong = []
    for s, t, vc, vn in test_loader:
        x_out, x = layer(s, vc, vn)
        for i in range(x.shape[0]):
            loss = nn.MSELoss(reduction="mean")(x_out[i], x[i])
            # stdtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t[0][-1][0].data.item()))
            # print(stdtime, loss.data.item())
            loss_seq.append(loss.data.item())
            if loss > test_args["threshold"]:
                sth_wrong.append(1)
            else:
                sth_wrong.append(0)

    if not os.path.exists('./return/epoch{}/{}/'.format(modelnum, severname)):
        os.makedirs('./return/epoch{}/{}/'.format(modelnum, severname))
    f = open("./return/epoch{}/{}/AEtest.csv".format(modelnum, severname), "w")
    writer = csv.writer(f)
    for i in loss_seq:
        writer.writerow((str(i),))
    f.close()
    print("        {} anomalies detected in {} seqs.".format(sum(sth_wrong), len(sth_wrong)))
    f = open("./return/epoch{}/{}/ans.txt".format(modelnum, severname), "a")
    f.write("AE:\n" + str(sth_wrong) + "\n")
    f.close()

