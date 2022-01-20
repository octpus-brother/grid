import torch
import hashlib

hashnum = 31

categorical_dic = [0, 0, 0, 0, 2, 0, 0, hashnum, 0, 0, 0, 2,

                   0, 0, 0, 0, 0, 3, 3, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
                   3, 3, 0, 2, 0]

numerical_dic = [1, 2, 1, 2, 0, 1, 1, 0, 1, 1, 3, 0,

                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0]

class_num = len(categorical_dic) + 1  # 42
categorical_num = sum(categorical_dic)  # 63
numerical_num = sum(numerical_dic)  # 17


def get_tensors(seq, vc_dic, vn_dic):
    s = torch.zeros(class_num)
    vc = torch.zeros(categorical_num)
    vn = torch.zeros(numerical_num)
    s[seq] = 1
    cseq = sum(categorical_dic[:seq])
    nseq = sum(numerical_dic[:seq])
    for i in vc_dic:
        vc[cseq + i] = 1
    for i in vn_dic:
        vn[nseq + i] = vn_dic[i]
    return s.view(-1, 1), vc.view(-1, 1), vn.view(-1, 1)


def svr41(content):  # 0,1
    seq = 0
    num = float(content.strip().strip("%")) / 100.0
    return seq, {}, {0: num}


def svr42(content):  # 0,2
    seq = 1
    content = content.strip().split()
    cat = int(content[0])
    num = float(content[1].strip("%")) / 100.0
    if cat == 1:
        return seq, {}, {0: num}
    else:
        return seq, {}, {1: num}


def svr43(content):  # 0,1
    seq = 2
    num = float(content.strip().strip("%")) / 100.0
    return seq, {}, {0: num}


def svr44(content):  # 0,2
    seq = 3
    content = content.strip().split()
    receive = 0
    send = 0
    for i in range(int(len(content)/4)):
        receive += int(content[4 * i + 2])
        send += int(content[4 * i + 3])
    receive /= 1000000.0
    send /= 1000000.0
    return seq, {}, {0: receive, 1: send}


def svr45(content):  # 2,0
    seq = 4
    choice = int(content[0])
    return seq, {choice: 1}, {}


def svr411(content):  # 0,1
    seq = 5
    num = int(content[0])
    return seq, {}, {0: num}


def svr412(content):  # 0,1
    seq = 6
    num = int(content[0])
    return seq, {}, {0: num}


def svr413(content):  # 31,0
    seq = 7
    content = content.strip().split()
    content = content[0] + " " + content[2]
    ans = hashlib.md5(content.encode()).hexdigest()
    ans = int(ans, 16)
    ans = ans % hashnum
    return seq, {ans: 1}, {}


def svr415(content):  # 0,1
    seq = 8
    content = content.strip().split()
    content = content[1::2]
    for i in range(len(content)):
        content[i] = int(content[i])
    num = max(content)
    return seq, {}, {0: num}


def svr416(content):  # 0,1
    seq = 9
    content = content.strip().split()
    content = content[1::2]
    for i in range(len(content)):
        content[i] = int(content[i])
    num = min(content)
    return seq, {}, {0: num}


def svr417(content):  # 0,3
    seq = 10
    content = content.strip().split()
    return seq, {}, {0: int(content[0]), 1: int(content[1]), 2: int(content[2])}


def svr418(content):  # 2,0
    seq = 11
    state = int(content[0])
    return seq, {state: 1}, {}


def svr512(content):  # 0,0
    seq = 12
    content = content.strip().split()
    return seq, {}, {}


def svr513(content):  # 0,0
    seq = 13
    content = content.strip().split()
    return seq, {}, {}


def svr514(content):  # 0,0
    seq = 14
    content = content.strip().split()
    return seq, {}, {}


def svr515(content):  # 0,0
    seq = 15
    content = content.strip().split()
    return seq, {}, {}


def svr516(content):  # 0,0
    seq = 16
    content = content.strip().split()
    return seq, {}, {}


def svr517(content):  # 3,0
    seq = 17
    content = content.strip().split("{")
    choice = int(content[-1][0])
    return seq, {choice: 1}, {}


def svr518(content):  # 3,0
    seq = 18
    content = content.strip().split("{")
    choice = int(content[-1][0])
    return seq, {choice: 1}, {}


def svr519(content):  # 2,0
    seq = 19
    content = content.strip().split()
    state = int(content[0])
    return seq, {state: 1}, {}


def svr520(content):  # 2,0
    seq = 20
    content = content.strip().split()
    state = int(content[0])
    return seq, {state: 1}, {}


def svr521(content):  # 2,0
    seq = 21
    content = content.strip().split()
    state = int(content[0])
    return seq, {state: 1}, {}


def svr522(content):  # 2,0
    seq = 22
    content = content.strip().split()
    state = int(content[0])
    return seq, {state: 1}, {}


def svr523(content):  # 2,0
    seq = 23
    content = content.strip().split()
    state = int(content[0])
    return seq, {state: 1}, {}


def svr524(content):  # 2,0
    seq = 24
    content = content.strip().split()
    state = int(content[0])
    return seq, {state: 1}, {}


def svr525(content):  # 0,4
    seq = 25
    content = content.strip().split()
    return seq, {}, {}


def svr526(content):  # 0,0
    seq = 26
    content = content.strip().split()
    return seq, {}, {}


def svr527(content):  # 0,0
    seq = 27
    content = content.strip().split()
    return seq, {}, {}


def svr528(content):  # 0,0
    seq = 28
    return seq, {}, {}


def svr529(content):  # 0,0
    seq = 29
    return seq, {}, {}


def svr530(content):  # 0,0
    seq = 30
    return seq, {}, {}


def svr531(content):  # 0,0
    seq = 31
    return seq, {}, {}


def svr532(content):  # 0,0
    seq = 32
    return seq, {}, {}


def svr534(content):  # 0,0
    seq = 34
    return seq, {}, {}


def svr535(content):  # 0,0
    seq = 35
    return seq, {}, {}


def svr536(content):  # 2,0
    seq = 36
    state = int(content[0])
    return seq, {state: 1}, {}


def svr537(content):  # 3,0
    seq = 37
    content = content.strip().split("{")
    choice = int(content[-1][0])
    return seq, {choice: 1}, {}


def svr538(content):  # 3,0
    seq = 38
    content = content.strip().split("{")
    choice = int(content[-1][0])
    return seq, {choice: 1}, {}


def svr539(content):  # 0,0
    seq = 39
    s = torch.zeros(class_num)
    vc = torch.zeros(categorical_num)
    vn = torch.zeros(numerical_num)
    return seq, {}, {}


def svr540(content):  # 2,0
    seq = 40
    content = content.strip()
    state = int(content[-1])
    return seq, {state: 1}, {}


def svrother(content):  # 0,0
    seq = 41
    return seq, {}, {}

func = {}
func["SVR4 1"] = svr41
func["SVR4 2"] = svr42
func["SVR4 3"] = svr43
func["SVR4 4"] = svr44
func["SVR4 5"] = svr45
func["SVR4 11"] = svr411
func["SVR4 12"] = svr412
func["SVR4 13"] = svr413
func["SVR4 15"] = svr415
func["SVR4 16"] = svr416
func["SVR4 17"] = svr417
func["SVR4 18"] = svr418
func["SVR5 12"] = svr512
func["SVR5 13"] = svr513
func["SVR5 14"] = svr514
func["SVR5 15"] = svr515
func["SVR5 16"] = svr516
func["SVR5 17"] = svr517
func["SVR5 18"] = svr518
func["SVR5 19"] = svr519
func["SVR5 20"] = svr520
func["SVR5 21"] = svr521
func["SVR5 22"] = svr522
func["SVR5 23"] = svr523
func["SVR5 24"] = svr524
func["SVR5 25"] = svr525
func["SVR5 26"] = svr526
func["SVR5 27"] = svr527
func["SVR5 30"] = svr530
func["SVR5 31"] = svr531
func["SVR5 32"] = svr532
func["SVR5 34"] = svr534
func["SVR5 35"] = svr535
func["SVR5 36"] = svr536
func["SVR5 37"] = svr537
func["SVR5 38"] = svr538
func["SVR5 39"] = svr539
func["SVR5 40"] = svr540
