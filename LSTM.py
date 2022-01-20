import torch
import torch.nn as nn
from torch import optim
import csv
import pickle
import time
import os
from torch.utils.data import Dataset, DataLoader
import getdata as gd
from getdata import SeqDataset
import warnings
warnings.filterwarnings("ignore")


torch.set_default_tensor_type(torch.DoubleTensor)


class LSTM(nn.Module):
    def __init__(self, Dim, Embedding_sz, Output_sz, Event_num, Categorical_num, Numerical_num, Device):
        super().__init__()
        self.Dim = Dim
        self.Embedding_sz = Embedding_sz
        self.Output_sz = Output_sz
        self.Device = Device
        self.Wx = nn.Parameter(torch.Tensor(Embedding_sz, Output_sz * 4))
        self.Wh = nn.Parameter(torch.Tensor(Output_sz, Output_sz * 4))
        self.Wc = nn.Parameter(torch.Tensor(3, Output_sz))
        self.bias = nn.Parameter(torch.Tensor(Output_sz * 4))
        self.Ve = nn.Parameter(torch.Tensor(Event_num, Embedding_sz))
        self.Vc = nn.Parameter(torch.Tensor(Categorical_num, Embedding_sz))
        self.Vn = nn.Parameter(torch.Tensor(Numerical_num, Embedding_sz))
        self.linear = nn.Linear(Output_sz, Dim)
        self.init_weights()

    def init_weights(self):
        for weight in self.parameters():
            weight.data.uniform_(-1, 1)

    def forward(self, event, time, vc, vn, init_states=None):
        """Assumes inputs is of shape (batch, sequence, features)"""
        """features = (event, time, valuec, valuen)"""
        bs, seq_sz, _ = event.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.Output_sz).to(self.Device),
                        torch.zeros(bs, self.Output_sz).to(self.Device))
        else:
            h_t, c_t = init_states
        HS = self.Output_sz
        for t in range(seq_sz):
            event_t = event[:, t, :]
            # time_t = time[:, t, :]
            vc_t = vc[:, t, :]
            vn_t = vn[:, t, :]

            s = event_t @ self.Ve
            x_t = s + 2.0 * (vc_t @ self.Vc + torch.tanh(vn_t @ self.Vn))
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.Wx + h_t @ self.Wh + self.bias
            j_t = 1
            g_t = torch.tanh(gates[:, HS * 2:HS * 3])
            i_t, f_t, o_t = (torch.sigmoid(gates[:, :HS] + c_t * self.Wc[0]),  # input
                             torch.sigmoid(gates[:, HS:HS * 2] + c_t * self.Wc[1]),  # forget
                             torch.sigmoid(gates[:, HS * 3:] + c_t * self.Wc[2]),  # output
                             )
            c_t_hat = f_t * c_t + i_t * g_t
            c_t = j_t * c_t_hat + (1 - j_t) * c_t
            h_t_hat = o_t * torch.tanh(c_t_hat)
            h_t = j_t * h_t_hat + (1 - j_t) * h_t
            hidden_seq.append(h_t.unsqueeze(0))
        # hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        # hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        y_t = self.linear(h_t)
        return y_t


def train(args, device, trainlogs):
    layer = LSTM(args["Dim"], args["Embedding_sz"], args["Output_sz"], args["class_num"],
                 args["categorical_num"], args["numerical_num"], device).to(device)
    optimizer = optim.Adam(layer.parameters(), lr=args["LearningRate"], weight_decay=args["WeightDecay"])

    s_seq, time_seq, vc_seq, vn_seq = gd.log2tensors(trainlogs, device)
    data_s = gd.split_seq(s_seq, args["WindowSize"], args["Stride"])
    data_t = gd.split_seq(time_seq, args["WindowSize"], args["Stride"])
    data_vc = gd.split_seq(vc_seq, args["WindowSize"], args["Stride"])
    data_vn = gd.split_seq(vn_seq, args["WindowSize"], args["Stride"])
    with torch.no_grad():
        center = layer(s_seq, time_seq, vc_seq, vn_seq)
    # print("center: {}".format(center.data))

    train_data = SeqDataset(data_s, data_t, data_vc, data_vn)
    train_loader = DataLoader(train_data, batch_size=args["BatchSize"], shuffle=True)

    layer.train()
    loss_record = []
    for epoch in range(args["Epoches"]):
        loss_record.append("epoch:" + str(epoch))
        count = 0
        count_loss = 0
        for s, t, vc, vn in train_loader:
            count += 1
            optimizer.zero_grad()
            out = layer(s, t, vc, vn)
            loss = torch.sum((out - center) ** 2, dim=1).mean()
            if count % 100 == 1:
                progress = epoch / args["Epoches"] + count / (len(train_loader) * args["Epoches"])
                print("        {:.4f}\t{:.2%}".format(loss.data, progress))
            # else:
            #     print("{:.4f}".format(loss.data))
            loss_record.append(str(loss.data))
            if loss.data > args["LossThreshold"]:
                loss.backward()
                optimizer.step()
                count_loss += 1
        if count_loss / count < args["TrainThreshold"]:
            print("        Stop early.")
            break

    modelnum = args["modelnum"]
    severname = args["severname"]
    if not os.path.exists('./models/LSTM/epoch{}/{}/'.format(modelnum, severname)):
        os.makedirs('./models/LSTM/epoch{}/{}/'.format(modelnum, severname))
    torch.save(layer.state_dict(), './models/LSTM/epoch{}/{}/LSTMmodel.pth'.format(modelnum, severname))
    torch.save(optimizer.state_dict(), './models/LSTM/epoch{}/{}/LSTMoptim.pth'.format(modelnum, severname))
    f = open('./models/LSTM/epoch{}/{}/LSTMcenter&args.pkl'.format(modelnum, severname), 'wb')
    pickle.dump(center, f)
    pickle.dump(args, f)
    f.close()
    f = open("./models/LSTM/epoch{}/{}/LSTMtrain.csv".format(modelnum, severname), "w")
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
    f = open('./models/LSTM/epoch{}/{}/LSTMcenter&args.pkl'.format(modelnum, severname), 'rb')
    center = pickle.load(f).to(device)
    args = pickle.load(f)
    f.close()

    layer = LSTM(args["Dim"], args["Embedding_sz"], args["Output_sz"], args["class_num"],
                 args["categorical_num"], args["numerical_num"], device).to(device)

    layer.load_state_dict(torch.load('./models/LSTM/epoch{}/{}/LSTMmodel.pth'.format(modelnum, severname)))
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
        out = layer(s, t, vc, vn)
        loss = torch.sum((out - center) ** 2, dim=1)
        # stdtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t[0][-1][0].data.item()))
        # print(stdtime, loss.data.item())
        cur_loss = loss.detach().numpy().tolist()
        loss_seq += cur_loss
        for i in cur_loss:
            if i > test_args["TestThreshold"] * args["LossThreshold"]:
                sth_wrong.append(1)
            else:
                sth_wrong.append(0)

    if not os.path.exists('./return/epoch{}/{}/'.format(modelnum, severname)):
        os.makedirs('./return/epoch{}/{}/'.format(modelnum, severname))
    f = open("./return/epoch{}/{}/LSTMtest.csv".format(modelnum, severname), "w")
    writer = csv.writer(f)
    for i in loss_seq:
        writer.writerow((str(i),))
    f.close()
    print("        {} anomalies detected in {} seqs.".format(sum(sth_wrong), len(sth_wrong)))
    f = open("./return/epoch{}/{}/ans.txt".format(modelnum, severname), "a")
    f.write("LSTM:\n" + str(sth_wrong) + "\n")
    f.close()

