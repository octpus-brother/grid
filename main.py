import getdata as gd
import utils
import sys
import os
import traceback

kargs = sys.argv
ip_port = kargs[1]
topic = kargs[2]
group = kargs[3]
start_epoch = int(kargs[4])
end_epoch = int(kargs[5])
listen_time = int(kargs[6])
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
if not os.path.exists('./return/'):
    os.makedirs('./return/')
mylog = open("./return/mylog.txt", "a")
mylog.write("---------------------------------------------------------------\n")
mylog.write(utils.get_time() + " Start running\n")

for epoch in range(start_epoch, end_epoch):
    print("Round {} in [{}, {})".format(epoch, start_epoch, end_epoch))
    mylog.write("  " + utils.get_time() + " Round {} in [{}, {})\n".format(epoch, start_epoch, end_epoch))
    if not os.path.exists('./data/'):
        os.makedirs('./data/')
    logpath = './data/epoch{}.txt'.format(epoch)
    try:
        gd.readfromkafka(ip_port, topic, group, epoch, logpath, listen_time)
        mylog.write("  " + utils.get_time() + " epoch: " + str(epoch) + " reading kafka: done\n")
    except Exception:
        print("Something wrong happens while consuming on kafka")
        traceback.print_exc()
        mylog.write("  " + utils.get_time() + " epoch: " + str(epoch) + " reading kafka: error\n")
        mylog.write(traceback.format_exc() + "\nstop\n\n\n\n\n")
        break

    try:
        train_data, test_data = gd.analysingfile(logpath)
        mylog.write("  " + utils.get_time() + " epoch: " + str(epoch) +
                    "  analysing file: done, {} severs to train\n".format(len(train_data)))
    except Exception:
        print("Something wrong happens while analysing log file")
        traceback.print_exc()
        mylog.write("  " + utils.get_time() + " epoch: " + str(epoch) + "  analysing file: error\n")
        mylog.write(traceback.format_exc() + "\nstop\n\n\n\n\n")
        break

    if len(train_data) < 4:
        listen_time *= 1.5
    print("{} usable severs found.".format(len(train_data)))
    if len(train_data) == 0:
        print("No usable severs found. New round begins.")
        continue

    for sever in train_data:
        if not os.path.exists('./return/epoch{}/{}/'.format(epoch, sever)):
            os.makedirs('./return/epoch{}/{}/'.format(epoch, sever))
        ans_path = './return/epoch{}/{}/ans.txt'.format(epoch, sever)
        print("    For sever {}:".format(sever))
        mylog.write("    " + utils.get_time() + " epoch: " + str(epoch) + " sever: " + sever + "\n")
        length = len(train_data[sever].record) + len(test_data[sever].record)
        print("    " + str(length) + " logs")
        mylog.write("    " + utils.get_time() + " " + str(length) + " logs\n")

        print("    Getting baseline on {}".format(sever))
        try:
            utils.getbaseline(test_data[sever], ans_path)
            print("    Getting baseline: done")
        except Exception:
            print("Something wrong happens while reading baseline on " + sever)
            traceback.print_exc()
            mylog.write("      " + utils.get_time() + " epoch: " + str(epoch) +
                        " sever: " + sever + " getting baseline: error\n")
            mylog.write(traceback.format_exc() + "\n")
            print("      Skipped")
            continue

        for model in utils.modellist:
            print("      Train {}:".format(model))
            try:
                utils.trainer(train_data[sever], model, epoch, sever, device)
                mylog.write("      " + utils.get_time() + " epoch: " + str(epoch) +
                            " sever: " + sever + " model: " + model + " training: done\n")
            except Exception:
                print("Something wrong happens while training " + model + " on " + sever)
                traceback.print_exc()
                mylog.write("      " + utils.get_time() + " epoch: " + str(epoch) +
                            " sever: " + sever + " model: " + model + " training: error\n")
                mylog.write(traceback.format_exc() + "\n")
                print("      Skipped")
                continue
            print("      Test {}:".format(model))
            try:
                utils.tester(test_data[sever], model, epoch, sever, device)
                mylog.write("      " + utils.get_time() + " epoch: " + str(epoch) +
                            " sever: " + sever + " model: " + model + " testing: done\n")
            except Exception:
                print("Something wrong happens while testing " + model + " on " + sever)
                traceback.print_exc()
                mylog.write("      " + utils.get_time() + " epoch: " + str(epoch) +
                            " sever: " + sever + " model: " + model + " testing: error\n")
                mylog.write(traceback.format_exc() + "\n")
                print("      Skipped")
                continue
        print("\n")
        mylog.write("\n")
    print("\n")
    mylog.write("\n\n\n")
mylog.close()

# filepath = "/home/grid/haveatry/11.22.2.92#d2sta22.txt"
# svrname = "11.22.2.92#d2sta22"
#
# modelname = "LSTM"  # HTSAD, deeplog, PCArecon, AE, LSTM
#
# testfilepath = "/home/grid/haveatry/tests/test1zhujireal_0.txt"
