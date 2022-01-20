from kafka import KafkaConsumer
import time
import datetime
import preprocess as pp
import torch
from torch.utils.data import Dataset


def readfromkafka(ip_port, topic, group, epoch, logpath, listen_time=2000):

    f = open(logpath, "a")
    groupid = group + str(epoch)
    consumer = KafkaConsumer(bootstrap_servers=ip_port,
                             group_id=groupid,
                             auto_offset_reset="latest")
    consumer.subscribe(topics=[topic])
    time_start = time.time()
    time_now = time_start
    while time_now - time_start < listen_time:
        time.sleep(10)
        msg = consumer.poll(timeout_ms=10)
        consumer.commit()
        time_now = time.time()
        for tp in msg:
            for cr in msg[tp]:
                f.write("{}\n".format(cr.value.decode()))

    f.close()


class Record:
    def __init__(self, level, time, device, devicecode, typeid, subtypeid, content):
        self.level = level
        self.time = time
        self.device = device
        self.devicecode = devicecode
        self.typeid = typeid
        self.subtypeid = subtypeid
        self.content = content

    def show(self):
        print(self.level, self.time, self.device, self.devicecode, self.typeid, self.subtypeid, self.content)

    def trans(self):
        timestr = self.time
        devicecode = self.devicecode
        typeid = self.typeid
        subtypeid = self.subtypeid
        content = self.content
        datetime_obj = datetime.datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S")
        timestamp = time.mktime(datetime_obj.timetuple())
        funcname = devicecode + str(typeid) + " " + str(subtypeid)
        if funcname in pp.func.keys():
            return timestamp, pp.func[funcname](content)
        else:
            return timestamp, pp.svrother(content)


def readlogline(line):
    line = line.strip().strip(b'\x00'.decode())
    data = line.split(" ")
    # level = int(data[0])
    level = int(data[0][1])
    time = data[1] + " " + data[2]
    device = data[3]
    devicecode = data[4]
    typeid = int(data[5])
    subtypeid = int(data[6])
    content = ""
    for i in range(7, len(data)):
        content += data[i] + " "
    record = Record(level, time, device, devicecode, typeid, subtypeid, content)
    return record


class Sever:
    def __init__(self, device):
        self.device = device
        self.record = []

    def insert(self, record):
        self.record.append(record)

    def show(self):
        for item in self.record:
            item.show()


def analysingfile(logfile, min_length=4000, max_length=10000):
    with open(logfile, "r") as f:
        done = 0
        severs = {}
        while not done:
            line = f.readline()
            if line == "":
                done = 1
            else:
                currecord = readlogline(line)
                if currecord.device not in severs:
                    severs[currecord.device] = Sever(currecord.device)
                severs[currecord.device].insert(currecord)
        train_data = {}
        test_data = {}
        for name in severs:
            length = len(severs[name].record)
            if length < min_length:
                continue
            elif length > max_length:
                severs[name].record = severs[name].record[:max_length]
            train_data[name] = Sever(name)
            train_data[name].record = severs[name].record[:length//5]
            test_data[name] = Sever(name)
            test_data[name].record = severs[name].record[length//5:]
            if len(train_data) > 20:
                break
        return train_data, test_data


def log_processor(records):
    processed_log = []
    for item in records.record:
        processed_log.append(item.trans())
    return processed_log


def log2tensors(logs, device):
    processed_log = log_processor(logs)
    input_t = []
    input_s = []
    input_vc = []
    input_vn = []
    for item in processed_log:
        input_t.append(item[0])
        tensors = pp.get_tensors(item[1][0], item[1][1], item[1][2])
        input_s.append(tensors[0])
        input_vc.append(tensors[1])
        input_vn.append(tensors[2])

    # temp_timeseq = np.array(input_t)  # don't know why have to convert to array
    time_seq = torch.Tensor(input_t).view(-1, 1)
    s_seq = torch.cat(input_s, 1).transpose(0, 1).contiguous()
    vc_seq = torch.cat(input_vc, 1).transpose(0, 1).contiguous()
    vn_seq = torch.cat(input_vn, 1).transpose(0, 1).contiguous()

    s_seq = s_seq.view(1, -1, pp.class_num).to(device)
    time_seq = time_seq.view(1, -1, 1).to(device)
    vc_seq = vc_seq.view(1, -1, pp.categorical_num).to(device)
    vn_seq = vn_seq.view(1, -1, pp.numerical_num).to(device)

    return s_seq, time_seq, vc_seq, vn_seq


def read_log_keywise(logs):
    processed_log = log_processor(logs)
    length = len(processed_log)
    count = 0
    seqs_keywise = {}
    data_type = {}
    for item in processed_log:
        key = item[1][0]
        categorical_data = item[1][1]
        numerical_data = item[1][2]
        if key not in seqs_keywise:
            if len(categorical_data) != 0:
                data_type[key] = "categorical"
            elif len(numerical_data) != 0:
                data_type[key] = "numerical"
            else:
                data_type[key] = "empty"
            seqs_keywise[key] = [None for i in range(2 * length)]
            if data_type[key] is "categorical":
                for data in categorical_data:
                    seqs_keywise[key][count] = data
                    count += 1
            elif data_type[key] is "numerical":
                for data in numerical_data:
                    seqs_keywise[key][count] = numerical_data[data]
                    count += 1
            else:
                # seqs_keywise[key][count] = 0
                count += 1
        else:
            if data_type[key] is "categorical":
                for data in categorical_data:
                    seqs_keywise[key][count] = data
                    count += 1
            elif data_type[key] is "numerical":
                for data in numerical_data:
                    seqs_keywise[key][count] = numerical_data[data]
                    count += 1
            else:
                # seqs_keywise[key][count] = 0
                count += 1
    # for item in seqs_keywise:
    #     print(seqs_keywise[item])
    # print(data_type)
    return data_type, seqs_keywise


def split_seq(seq, window, stride):
    length = seq.shape[1]
    num = (length - window) // stride + 1
    # print(str(num) + " sequences")
    splited_seqs = []
    for i in range(num):
        pos = i * stride
        splited_seqs.append(seq[:, pos: window + pos, :])
    splited_seqs = torch.cat(splited_seqs)
    return splited_seqs


class SeqDataset(Dataset):
    def __init__(self, s, t, vc, vn):
        self.s_seq = s
        self.t_seq = t
        self.vc_seq = vc
        self.vn_seq = vn
        # print(str(s.shape[0]) + " sequences")

    def __getitem__(self, index):
        return self.s_seq[index], self.t_seq[index], self.vc_seq[index], self.vn_seq[index]

    def __len__(self):
        return self.s_seq.shape[0]