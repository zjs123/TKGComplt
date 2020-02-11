import os
import pickle
import argparse

class Data_process:
    def __init__(self, args):
        self.dataset = args.dataset
        path= "./dataset/"+args.dataset
        self.get_seq(path)
        self.get_train_triples()
        
        print("data process finshed")

    def get_seq(self,path):
        seq_withTime = {}
        seq_relation = {}
        print("-----Get seqs from " + path + "/-----")
        if os.path.exists(path+"/seq_withTime.pickle") and os.path.exists(path+"/r_seq.pickle"):
            seq_withTime_Input = open(path + "/seq_withTime.pickle", "rb")
            seq_relation_Input = open(path + "/r_seq.pickle", "rb")
            seq_withTime.update(pickle.load(seq_withTime_Input))
            seq_relation.update(pickle.load(seq_relation_Input))
            seq_withTime_Input.close()
            seq_relation_Input.close()
            return
        inputData=open(path+"/train.txt",encoding='utf-8')
        statData=open(path+"/entity2id.txt",encoding='utf-8')
        numOfEntity = int(statData.readline().strip().split()[0])
        lines=inputData.readlines()[1:]
        for i in range(numOfEntity):
            seq={}
            for line in lines:
                if len(line.strip().split())==1:
                    continue
                head=line.strip().split()[0]
                relation=line.strip().split()[1]
                time=line.strip().split()[3].split('-')[0]
                if(int(head)==i):
                    if '#' not in time and time!='':
                        seq[int(time)]=relation
                        #print(time)
            sort_seq=sorted(seq.items(),key=lambda d: d[0])
            seq_withTime[i]=sort_seq
            r_seq=[]
            for k in sort_seq:
                r_seq.append(int(k[1]))
            seq_relation[i]=r_seq
        inputData.close()
        self.write_seq(path,seq_withTime,seq_relation)
        
    def get_train_triples(self):
        print("-----Get train_data from " + self.dataset + "/-----")
        train_triple={}
        train_triple_reverse={}
        statData=open("./dataset/"+self.dataset+"/entity2id.txt",encoding='utf-8')
        numOfEntity = int(statData.readline().strip().split()[0])
        if os.path.exists("./dataset/"+self.dataset+"/train_ILP.pickle") :
            train_ILP = open("./dataset/"+self.dataset+"/train_ILP.pickle", "rb")
            train_triple=pickle.load(train_ILP)
            train_ILP.close()
        else:
            data_path="./dataset/"+self.dataset+"/train.txt"
            train_data=open(data_path,encoding='utf-8')
            lines=train_data.readlines()
            train_data.close()
            for i in range(numOfEntity):
                train_triple[i]={}
                relation_list=list()
                tail_list=list()
                time_list=list()
                for line in lines:
                    if len(line.strip().split())<3:
                        continue
                    head=line.strip().split()[0]
                    tail=line.strip().split()[2]
                    if int(head) != i:
                        continue
                    relation=int(line.strip().split()[1])
                    tail=int(line.strip().split()[2])
                    time_start=line.strip().split()[3].split('-')[0]
                    time_end=line.strip().split()[4].split('-')[0]
                    relation_list.append(relation)
                    tail_list.append(tail)
                    if time_start!='' and '#' not in time_start:
                        time_start = int(time_start)
                    else:
                        time_start=0
                    if time_end!='' and '#' not in time_end:
                        time_end=int(time_end)
                    else:
                        time_end=time_start+1
                    #time=line.strip().split()[3]
                    #if '#' not in time and time!='':
                    #    m=time.strip().split('-')[1]
                    #    d=time.strip().split('-')[2]
                    #    time_start=int(m)*100+int(d)
                    #    time_end=time_start+1
                    #else:
                    #    time_start=0
                    #    time_end=time_start+1
                    time_list.append([time_start,time_end])
                train_triple[i]['r']=list(relation_list)
                train_triple[i]['t']=list(tail_list)
                train_triple[i]['time']=list(time_list)
        self.write_triple(train_triple,"train_ILP")
            
        if os.path.exists("./dataset/"+self.dataset+"/train_ILP_reverse.pickle"):
            train_ILP_reverse= open("./dataset/"+self.dataset+"/train_ILP_reverse.pickle", "rb")
            train_triple_reverse=pickle.load(train_ILP_reverse)
            train_ILP_reverse.close()
        else:
            data_path="./dataset/"+self.dataset+"/train.txt"
            train_data=open(data_path,encoding='utf-8')
            lines=train_data.readlines()
            train_data.close()
            for i in range(numOfEntity):
                train_triple_reverse[i]={}
                relation_list=list()
                head_list=list()
                time_list=list()
                for line in lines:
                    if len(line.strip().split())<3:
                        continue
                    head=line.strip().split()[0]
                    tail=line.strip().split()[2]
                    if int(tail) != i:
                        continue
                    relation=int(line.strip().split()[1])
                    head=int(line.strip().split()[0])
                    time_start=line.strip().split()[3].split('-')[0]
                    time_end=line.strip().split()[4].split('-')[0]
                    relation_list.append(relation)
                    head_list.append(head)
                    if time_start!='' and '#' not in time_start:
                        time_start = int(time_start)
                    else:
                        time_start=0
                    if time_end!='' and '#' not in time_end:
                        time_end=int(time_end)
                    else:
                        time_end=time_start+1
                    time_list.append([time_start,time_end])
                train_triple_reverse[i]['r']=list(relation_list)
                train_triple_reverse[i]['h']=list(head_list)
                train_triple_reverse[i]['time']=list(time_list)
        self.write_triple(train_triple_reverse,"train_ILP_reverse")

    def write_seq(self,path,seq_withTime, seq_relation):
        #print "-----Writing Training Results to " + self.outAdd + "-----"
        seq_withTime_Add = path + "/seq_withtime.pickle"
        r_seq_Add = path + "/r_seq.pickle"
        Output_1 = open(seq_withTime_Add, "wb")
        Output_2 = open(r_seq_Add, "wb")
        pickle.dump(seq_withTime, Output_1)
        pickle.dump(seq_relation,Output_2)
        Output_1.close()
        Output_2.close()
        
    def write_triple(self,train_triples,path):
        train_ILP= "./dataset/"+self.dataset+"/"+path+".pickle"
        Output = open(train_ILP, "wb")
        pickle.dump(train_triples, Output)
        Output.close() 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data_process")
    parser.add_argument("--dataset",dest="dataset",type=str,default="YAGO11K")
    args=parser.parse_args()
    Data_process(args)
