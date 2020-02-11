import re
import os
import torch
import pickle
import copy
from collections import defaultdict as ddict

class readData:
    def __init__(self, inAdd, train2id, seq_withtime, seq_r, headRelation2Tail, tailRelation2Head, headTail2Relation,nums):
        self.inAdd = inAdd
        self.train2id = train2id
        self.seq_withtime = seq_withtime
        self.seq_r = seq_r
        self.headRelation2Tail = headRelation2Tail
        self.tailRelation2Head = tailRelation2Head
        self.headTail2Relation= headTail2Relation
        self.nums = nums
        self.numOfTriple = 0
        self.numOfEntity = 0
        self.numOfRelation = 0

        self.readTriple2id()

        self.readTrain2id()
        print ("number of triples: " + str(self.numOfTrainTriple))

        self.readEntity2id()
        print ("number of entities: " + str(self.numOfEntity))

        self.readRelation2id()
        print ("number of relations: " + str(self.numOfRelation))

        self.read_seq()
        print("Seq read finished")


        self.nums[0] = self.numOfTrainTriple
        self.nums[1] = self.numOfEntity
        self.nums[2] = self.numOfRelation

        # print self.numOfTriple
        # print self.train2id
        # print self.numOfEntity
        # print self.entity2id
        # print self.id2entity
        # print self.numOfRelation
        # print self.relation2id
        # print self.id2relation
        # print self.headRelation2Tail
        # print self.tailRelation2Head
    
    def readTriple2id(self):
        print ("-----Reading triple2id.txt from " + self.inAdd + "/-----")
        count = 0
        inputData = open(self.inAdd + "/train.txt",encoding='utf-8')
        line = inputData.readline()
        self.numOfTriple = int(re.findall(r"\d+", line)[0])
        print("num of triple= "+str(self.numOfTriple))
        #self.train2id["time"] = []
        line = inputData.readline()
        while line and line not in ["\n", "\r\n", "\r"]:
            reR = re.findall(r"\d+", line)
            if reR:
                tmpHead = int(line.strip().split()[0])
                #tmpTail = int(re.findall(r"\d+", line)[1])
                #tmpRelation = int(re.findall(r"\d+", line)[2])
                tmpTail = int(line.strip().split()[2])
                tmpRelation = int(line.strip().split()[1])
                #tmpTime = int(re.findall(r"\d+", line)[3])
                #self.train2id["time"].append(tmpTime)
                if tmpHead not in self.headRelation2Tail.keys():
                    self.headRelation2Tail[tmpHead] = {}
                    self.headRelation2Tail[tmpHead][tmpRelation] = []
                    self.headRelation2Tail[tmpHead][tmpRelation].append(tmpTail)
                else:
                    if tmpRelation not in self.headRelation2Tail[tmpHead].keys():
                        self.headRelation2Tail[tmpHead][tmpRelation] = []
                        self.headRelation2Tail[tmpHead][tmpRelation].append(tmpTail)
                    else:
                        if tmpTail not in self.headRelation2Tail[tmpHead][tmpRelation]:
                            self.headRelation2Tail[tmpHead][tmpRelation].append(tmpTail)
                if tmpTail not in self.tailRelation2Head.keys():
                    self.tailRelation2Head[tmpTail] = {}
                    self.tailRelation2Head[tmpTail][tmpRelation] = []
                    self.tailRelation2Head[tmpTail][tmpRelation].append(tmpHead)
                else:
                    if tmpRelation not in self.tailRelation2Head[tmpTail].keys():
                        self.tailRelation2Head[tmpTail][tmpRelation] = []
                        self.tailRelation2Head[tmpTail][tmpRelation].append(tmpHead)
                    else:
                        if tmpHead not in self.tailRelation2Head[tmpTail][tmpRelation]:
                            self.tailRelation2Head[tmpTail][tmpRelation].append(tmpHead)
                if tmpHead not in self.headTail2Relation.keys():
                    self.headTail2Relation[tmpHead] = {}
                    self.headTail2Relation[tmpHead][tmpTail] = []
                    self.headTail2Relation[tmpHead][tmpTail].append(tmpRelation)
                else:
                    if tmpTail not in self.headTail2Relation[tmpHead].keys():
                        self.headTail2Relation[tmpHead][tmpTail] = []
                        self.headTail2Relation[tmpHead][tmpTail].append(tmpRelation)
                    else:
                        if tmpRelation not in self.headTail2Relation[tmpHead][tmpTail]:
                            self.headTail2Relation[tmpHead][tmpTail].append(tmpRelation)
                count += 1
                line = inputData.readline()
            else:
                print ("error in triple2id.txt at Line " + str(count + 2))
                line = inputData.readline()
        inputData.close()
        if count == self.numOfTriple:
            return
        else:
            print ("error in triple2id.txt  count= "+str(count))
            return

    def readTrain2id(self):
        print ("-----Reading train2id.txt from " + self.inAdd + "/-----")
        count = 0
        inputData = open(self.inAdd + "/train.txt",encoding='utf-8')
        line = inputData.readline()
        self.numOfTrainTriple = int(re.findall(r"\d+", line)[0])
        self.train2id["h"] = []
        self.train2id["t"] = []
        self.train2id["r"] = []
        #self.train2id["time"] = []
        line = inputData.readline()
        while line and line not in ["\n", "\r\n", "\r"]:
            reR = re.findall(r"\d+", line)
            if reR:
                tmpHead = int(re.findall(r"\d+", line)[0])
                #tmpTail = int(re.findall(r"\d+", line)[1])
                #tmpRelation = int(re.findall(r"\d+", line)[2])
                tmpTail = int(re.findall(r"\d+", line)[2])
                tmpRelation = int(re.findall(r"\d+", line)[1])
                #tmpTime = int(re.findall(r"\d+", line)[3])
                self.train2id["h"].append(tmpHead)
                self.train2id["t"].append(tmpTail)
                self.train2id["r"].append(tmpRelation)
                #self.train2id["time"].append(tmpTime)
                count += 1
                line = inputData.readline()
            else:
                print ("error in train2id.txt at Line " + str(count + 2))
                line = inputData.readline()
        inputData.close()
        if count == self.numOfTrainTriple:
            return
        else:
            print ("error in train2id.txt")
            return

    def read_seq(self):
        print("-----Reading seqs from " + self.inAdd + "/-----")
        if os.path.exists(self.inAdd+"/seq_withtime.pickle") and os.path.exists(self.inAdd+"/r_seq.pickle"):
            seq_withtime_Input = open(self.inAdd + "/seq_withtime.pickle", "rb")
            seq_r_Input = open(self.inAdd + "/r_seq.pickle", "rb")
            self.seq_withtime.update(pickle.load(seq_withtime_Input))
            self.seq_r.update(pickle.load(seq_r_Input))
            seq_withtime_Input.close()
            seq_r_Input.close()
            return
        inputData=open(self.inAdd+"/train.txt",encoding='utf-8')
        lines=inputData.readlines()[1:]
        for i in range(self.numOfEntity):
            seq={}
            for line in lines:
                if len(line.strip().split())==1:
                    continue
                head=line.strip().split()[0]
                relation=line.strip().split()[1]
                time=line.strip().split()[3]#.split('-')[0]
                if(int(head)==i):
                    if '#' not in time and time!='':
                        seq[int(time)]=relation
                        #print(time)
            sort_seq=sorted(seq.items(),key=lambda d: d[0])
            self.seq_withtime[i]=sort_seq
            r_seq=[]
            for k in sort_seq:
                r_seq.append(int(k[1]))
            self.seq_r[i]=r_seq
        inputData.close()
        self.write()



    def readEntity2id(self):
        print ("-----Reading entity2id.txt from " + self.inAdd + "/-----")
        inputData = open(self.inAdd + "/entity2id.txt",encoding='utf-8')
        line = inputData.readline()
        self.numOfEntity = int(re.findall(r"\d+", line)[0])
        return


    def readRelation2id(self):
        print ("-----Reading relation2id.txt from " + self.inAdd + "/-----")
        inputData = open(self.inAdd + "/relation2id.txt",encoding='utf-8')
        line = inputData.readline()
        self.numOfRelation = int(re.findall(r"\d+", line)[0])
        return

    def write(self):
        #print "-----Writing Training Results to " + self.outAdd + "-----"
        seq_withtime_Add = self.inAdd + "/seq_withtime.pickle"
        r_seq_Add = self.inAdd + "/r_seq.pickle"
        Output_1 = open(seq_withtime_Add, "wb")
        Output_2 = open(r_seq_Add, "wb")
        pickle.dump(self.seq_withtime, Output_1)
        pickle.dump(self.seq_r,Output_2)
        Output_1.close()
        Output_2.close()