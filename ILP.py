# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 17:22:04 2020

@author: zjs
"""

import numpy as np
import argparse
import pickle
import torch
import time
import os
import re
import torch.nn as nn
class ILP:
    
    def __init__(self,args):
        self.testHead = []
        self.testRelation = []
        self.testTail = []
        self.testTime = []
        self.dataset = args.dataset
        self.Transmit = None
        self.preRead()
        self.norm = self.Transmit.norm
        self.numOfEntity = self.Transmit.numOfEntity
        self.numOfRelation = self.Transmit.numOfRelation
        self.numOfTestTriple = 0
        self.seq_withTime= {}
        self.seq_relation= {}
        self.train_triples, self.train_triples_reverse = self.read_train_triples()
        self.trainTriple_filter = None
        self.readTriple2id()
        self.creat_relation_set()
        self.read_seq()
        self.read_test_triples()
        
        self.ILP_solver()
        
    def readTriple2id(self):
        print ("-----Reading train2id.txt from " + self.dataset + "/-----")
        count = 0
        inputData = open("./dataset/"+self.dataset + "/train.txt",encoding='utf-8')
        line = inputData.readline()
        self.numOfTriple = int(re.findall(r"\d+", line)[0])
        #print("num of triple= "+str(self.numOfTriple))
        #self.train2id["time"] = []
        self.trainTriple_filter = torch.ones(self.numOfTriple, 4)
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
                self.trainTriple_filter[count, 0] = tmpHead
                self.trainTriple_filter[count, 1] = tmpRelation
                self.trainTriple_filter[count, 2] = tmpTail
                
                line = inputData.readline()
                count=count+1
        inputData.close()
        if count == self.numOfTriple:
            self.trainTriple_filter.long()
            return
        else:
            print ("error in triple2id.txt  count= "+str(count))
            return
    
    def read_test_triples(self):
        data_path="./dataset/"+self.dataset+"/test.txt"
        test_data=open(data_path,encoding='utf-8')
        lines=test_data.readlines()
        num=0
        for line in lines:
            if(len(line.strip().split())<5):
                continue
            num=num+1
            head=int(line.strip().split()[0])
            relation=int(line.strip().split()[1])
            tail=int(line.strip().split()[2])
            time_start=line.strip().split()[3].split('-')[0]
            time_end=line.strip().split()[4].split('-')[0]
            if time_start!='' and '#' not in time_start:
                time_start = int(time_start)
            else:
                time_start=0
            if time_end!='' and '#' not in time_end:
                time_end=int(time_end)
            else:
                time_end=time_start+1
            self.testHead.append(torch.tensor(head))
            self.testRelation.append(torch.tensor(relation))
            self.testTail.append(torch.tensor(tail))
            self.testTime.append(torch.tensor([time_start,time_end]))
            
        self.numOfTestTriple = num
            
    def read_seq(self):
        print("-----Reading seqs from " + self.dataset + "/-----")
        if os.path.exists("./dataset/"+self.dataset+"/seq_withtime.pickle") and os.path.exists("./dataset/"+self.dataset+"/r_seq.pickle"):
            seq_withTime_Input = open("./dataset/"+self.dataset + "/seq_withtime.pickle", "rb")
            seq_r_Input = open("./dataset/"+self.dataset + "/r_seq.pickle", "rb")
            self.seq_withTime.update(pickle.load(seq_withTime_Input))
            self.seq_relation.update(pickle.load(seq_r_Input))
            seq_withTime_Input.close()
            seq_r_Input.close()
            return
        inputData=open("./dataset/"+self.dataset+"/train.txt",encoding='utf-8')
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
            self.seq_withTime[i]=sort_seq
            r_seq=[]
            for k in sort_seq:
                r_seq.append(int(k[1]))
            self.seq_relation[i]=r_seq
        inputData.close()
    

    def read_train_triples(self):
        train_triple={}
        train_triple_reverse={}
        if os.path.exists("./dataset/"+self.dataset+"/train_ILP.pickle") :
            train_ILP = open("./dataset/"+self.dataset+"/train_ILP.pickle", "rb")
            train_triple=pickle.load(train_ILP)
            train_ILP.close()
        else:
            data_path="./dataset/"+self.dataset+"/train.txt"
            train_data=open(data_path,encoding='utf-8')
            lines=train_data.readlines()
            train_data.close()
            for i in range(self.numOfentity):
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
        self.write(train_triple,"train_ILP")
            
        if os.path.exists("./dataset/"+self.dataset+"/train_ILP_reverse.pickle"):
            train_ILP_reverse= open("./dataset/"+self.dataset+"/train_ILP_reverse.pickle", "rb")
            train_triple_reverse=pickle.load(train_ILP_reverse)
            train_ILP_reverse.close()
        else:
            data_path="./dataset/"+self.dataset+"/train.txt"
            train_data=open(data_path,encoding='utf-8')
            lines=train_data.readlines()
            train_data.close()
            for i in range(self.numOfentity):
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
        self.write(train_triple_reverse,"train_ILP_reverse")
        return train_triple ,train_triple_reverse
    
    def read_KG2vec(self,dataset):
        entityInput = open("./dataset/"+self.dataset+"/entity2vec_w.pickle", "rb")
        relationInput = open("./dataset/"+self.dataset+"/relation2vec_w.pickle", "rb")
        transInput = open("./dataset/"+self.dataset+"/trans2vec_w.pickle", "rb")
        tmpEntityEmbedding = pickle.load(entityInput)
        tmpRelationEmbedding = pickle.load(relationInput)
        tmpTransvec = pickle.load(transInput)
        entityInput.close()
        relationInput.close()
        transInput.close()
        return tmpEntityEmbedding, tmpRelationEmbedding, tmpTransvec
        
    def creat_relation_set(self):
        #YAGO11k
        if self.dataset == "YAGO11K":
            self.C_1=[4,6]#non-overlap relations 
            self.C_2_pre={7:[0,1,2,3,4,5,6,8,9],8:[0],9:[0],6:[0],5:[0,6],4:[0],3:[0,5,6],2:[0,9],1:[0,4]}#order relation r1 in (r1,r2) e.g. {works_at:born_in , graduated from}
            self.C_2_aft={0:[1,2,3,4,5,6,7,8,9],1:[7],2:[7],3:[7],4:[7,1],5:[7,3],6:[3,5,7,8],8:[7],9:[7,2]}#order relation r2 in (r1,r2) e.g. {born_in:works_at , died_in}
        
        #WIKI12k
        if self.dataset == "WIKI12K":
            self.C_1=[2, 3, 4, 7, 8, 13, 16, 19, 20]
            self.C_2_pre={2:[19],6:[9],7:[15],8:[4,12],9:[1,6,8,13,22,23],10:[21,22],12:[2],13:[0,6,9,15],14:[8,15,22,23],17:[0,15],19:[23],20:[4,23],21:[1,9,12,23],22:[23]}
            self.C_2_aft={0:[13,17],1:[9,21],2:[12],4:[8,20],6:[9,13],8:[9,14],9:[6,13,21],12:[8,21],13:[9],15:[7,13,14,17],19:[2],21:[10],22:[9,10,14],23:[9,14,19,20,21,22]}
        
        #WIKI36K
        if self.dataset == "WIKI36K":
            self.C_1=[17,19,32,35,43,55,58,59,62,64,73,76,77,86,88]
            self.C_2_pre={0:[6,27,29,30,53,69],1:[37,84],8:[26,34],9:[10,37,39,48],16:[29,53],18:[78],22:[93],23:[27,78],25:[36,81],26:[8],27:[18,31,69,87,93],29:[1,22,80],30:[22,23],33:[69],34:[6,52],36:[25,81],37:[56,62,87,90],38:[12,27,67,72],44:[70],45:[29,53],50:[6],54:[37,84],55:[54,87],60:[0,67],64:[22,29,30,78],65:[12,27,37,41,84],66:[6,30],67:[12,27,38],69:[31,80,86],70:[44],72:[12,38,47],76:[53],78:[27,69],80:[18],84:[1],85:[89],89:[39],90:[62],91:[34],92:[18,53],93:[78,94],94:[18]}
            self.C_2_aft={0:[60],1:[29,84],6:[0,34,50,66],8:[26],10:[9],12:[38,65,67,72],18:[27,80,92,94],22:[29,30,64],23:[30],25:[36],26:[8],27:[0,23,38,65,67,78],29:[0,16,45,64],30:[0,64,66],31:[27,69],34:[8,91],36:[25],37:[1,9,54,65],38:[67,72],39:[9,89],41:[65],44:[70],47:[72],48:[9],52:[34],53:[0,16,45,76,92],54:[55],56:[37],62:[37,90],67:[38,60],69:[0,27,33,78],70:[44],72:[38],78:[18,23,64,93],80:[29,69],81:[25,36],84:[1,54,65],86:[69],87:[27,37,55],89:[85],90:[37],93:[22,27],94:[93]}
        
        return 0
        
    def overlap_constrain(self,head,relation,tail,time):
        if relation not in self.C_1:
            return 1
        index_list=[i for i,v in enumerate(self.train_triples[head]['r']) if v==relation]
        for index in index_list:
            train_time=self.train_triples[head]['time'][index]
            target_time=time
            if train_time[1]<=target_time[0] or target_time[1]<=train_time[0]:
                continue
            else:
                return 0
        return 1
        
    def order_constrain(self,head,relation,tail,time,order_type):
        if(order_type == "pre"):
            if relation not in list(self.C_2_pre.keys()):
                return 1
            pre_relations=self.C_2_pre[relation]
            for i in range(len(self.train_triples[head]['r'])):
                if self.train_triples[head]['r'][i] in pre_relations:
                    if self.train_triples[head]['time'][i][0] <= time[0]:
                        continue
                    else:
                        return 0
            return 1
        
        if(order_type == "aft"):
            if relation not in list(self.C_2_aft.keys()):
                return 1
            aft_relations=self.C_2_aft[relation]
            for i in range(len(self.train_triples[head]['r'])):
                if self.train_triples[head]['r'][i] in aft_relations:
                    if self.train_triples[head]['time'][i][0] >= time[0]:
                        continue
                    else:
                        return 0
            return 1
        
    
    def overlap_constrain_h(self,head,relation,tail,time):
        if relation not in self.C_1:
            return 1
        index_list=[i for i,v in enumerate(self.train_triples[head]['r']) if v==relation]
        for index in index_list:
            train_time=self.train_triples[head]['time'][index]
            target_time=time
            if train_time[1]<=target_time[0] or target_time[1]<=train_time[0]:
                continue
            else:
                return 0
        return 1
        
    def overlap_constrain_t(self,head,relation,tail,time):
        if relation not in self.C_1:
            return 1
        index_list=[i for i,v in enumerate(self.train_triples_reverse[tail]['r']) if v==relation]
        for index in index_list:
            train_time=self.train_triples_reverse[tail]['time'][index]
            target_time=time
            if train_time[1]<=target_time[0] or target_time[1]<=train_time[0]:
                continue
            else:
                return 0
        return 1 
    def order_constrain_h(self,head,relation,tail,time):
        if relation in list(self.C_2_pre.keys()):
            pre_relations=self.C_2_pre[relation]
            for i in range(len(self.train_triples[head]['r'])):
                if self.train_triples[head]['r'][i] in pre_relations:
                    if self.train_triples[head]['time'][i][0] <= time[0]:
                        continue
                    else:
                        return 0
        if relation in list(self.C_2_aft.keys()):
            aft_relations=self.C_2_aft[relation]
            for i in range(len(self.train_triples[head]['r'])):
                if self.train_triples[head]['r'][i] in aft_relations:
                    if self.train_triples[head]['time'][i][0] >= time[0]:
                        continue
                    else:
                        return 0
        return 1
    def order_constrain_t(self,head,relation,tail,time):
        if relation in list(self.C_2_pre.keys()):
            pre_relations=self.C_2_pre[relation]
            for i in range(len(self.train_triples_reverse[tail]['r'])):
                if self.train_triples_reverse[tail]['r'][i] in pre_relations:
                    if self.train_triples_reverse[tail]['time'][i][0] <= time[0]:
                        continue
                    else:
                        return 0
        if relation in list(self.C_2_aft.keys()):
            aft_relations=self.C_2_aft[relation]
            for i in range(len(self.train_triples_reverse[tail]['r'])):
                if self.train_triples_reverse[tail]['r'][i] in aft_relations:
                    if self.train_triples_reverse[tail]['time'][i][0] >= time[0]:
                        continue
                    else:
                        return 0
        return 1
    
    
    def ILP_entity(self):
        mean_Rank_Raw=0.0
        mean_Rank_filter=0.0
        Hits10_Raw=0.0
        Hits10_filter=0.0
        for i in range(self.numOfTestTriple):
            self.testHeadEmbedding=self.Transmit.entity_embeddings(self.testHead[i])
            self.testRelationEmbedding=self.Transmit.relation_embeddings(self.testRelation[i])
            self.testTailEmbedding=self.Transmit.entity_embeddings(self.testTail[i])
            targetLoss = torch.norm(self.testHeadEmbedding+self.testRelationEmbedding-self.testTailEmbedding,self.norm).repeat(self.numOfEntity,1)
            C_1=self.overlap_constrain_h(int(self.testHead[i]),int(self.testRelation[i]),int(self.testTail[i]),self.testTime[i].numpy())
            C_2=self.order_constrain_h(int(self.testHead[i]),int(self.testRelation[i]),int(self.testTail[i]),self.testTime[i].numpy())
            if C_1*C_2 !=1 or float(targetLoss[0]) > 11:
                targetLoss=torch.FloatTensor([1000])
            tmpHeadEmbedding=self.testHeadEmbedding.repeat(self.numOfEntity,1)
            tmpRelationEmbedding=self.testRelationEmbedding.repeat(self.numOfEntity,1)
            tmpTailEmbedding=self.testTailEmbedding.repeat(self.numOfEntity,1)
            tmpHeadLoss=torch.norm(self.Transmit.entity_embeddings.weight.data + tmpRelationEmbedding - tmpTailEmbedding,
                                 self.norm, 1).view(-1, 1)
            tmpTailLoss = torch.norm(tmpHeadEmbedding + tmpRelationEmbedding - self.Transmit.entity_embeddings.weight.data,
                                 self.norm, 1).view(-1, 1)
            ILP_list_h=list()
            ILP_list_t=list()
            for j in range(self.numOfEntity):
                C_1_h=self.overlap_constrain_h(int(j),int(self.testRelation[i]),int(self.testTail[i]),self.testTime[i].numpy())
                C_2_h=self.order_constrain_h(int(j),int(self.testRelation[i]),int(self.testTail[i]),self.testTime[i].numpy())
                C_1_t=self.overlap_constrain_t(int(self.testHead[i]),int(self.testRelation[i]),int(j),self.testTime[i].numpy())
                C_2_t=self.order_constrain_t(int(self.testHead[i]),int(self.testRelation[i]),int(j),self.testTime[i].numpy())
                if C_1_h*C_2_h !=1 or float(tmpHeadLoss[j]) > 11:
                    ILP_list_h.append([10000])
                else:
                    ILP_list_h.append([0])
                if C_1_t*C_2_t !=1 or float(tmpTailLoss[j]) > 11:
                    ILP_list_t.append([10000])
                else:
                    ILP_list_t.append([0])
            ILP_tensor_h=torch.FloatTensor(ILP_list_h)
            ILP_tensor_t=torch.FloatTensor(ILP_list_t)
            tmpHeadLoss=tmpHeadLoss+ILP_tensor_h
            tmpTailLoss=tmpTailLoss+ILP_tensor_t
            wrongHead=torch.nonzero(nn.functional.relu(targetLoss - tmpHeadLoss))
            wrongTail=torch.nonzero(nn.functional.relu(targetLoss - tmpTailLoss))
            Rank_H=wrongHead.size()[0]+1
            Rank_T=wrongTail.size()[0]+1
            mean_Rank_Raw=mean_Rank_Raw+Rank_H+Rank_T
            if Rank_H<=10:
                Hits10_Raw=Hits10_Raw+1
            if Rank_T<=10:
                Hits10_Raw=Hits10_Raw+1
                
            numOfFilterHead=0
            numOfFilterTail=0
            for tmpWrongHead in wrongHead:
                numOfFilterHead += self.trainTriple_filter[(self.trainTriple_filter[:,0]==tmpWrongHead[0].float())&(self.trainTriple_filter[:,1]==self.testRelation[i].float())&(self.trainTriple_filter[:,2]==self.testTail[i].float())].size()[0]
            for tmpWrongTail in wrongTail:
                numOfFilterTail += self.trainTriple_filter[(self.trainTriple_filter[:,0]==self.testHead[i].float())&(self.trainTriple_filter[:,1]==self.testRelation[i].float())&(self.trainTriple_filter[:,2]==tmpWrongTail[0].float())].size()[0]
            
            Rank_H_filter=Rank_H-numOfFilterHead
            Rank_T_filter=Rank_T-numOfFilterTail
            
            mean_Rank_filter=mean_Rank_filter+Rank_H_filter+Rank_T_filter
            if Rank_H_filter<=10:
                Hits10_filter=Hits10_filter+1
            if Rank_T_filter<=10:
                Hits10_filter=Hits10_filter+1
            
        Hits10_Raw=Hits10_Raw/(2*self.numOfTestTriple)
        Hits10_filter=Hits10_filter/(2*self.numOfTestTriple)
        mean_Rank_Raw=mean_Rank_Raw/(2*self.numOfTestTriple)
        mean_Rank_filter=mean_Rank_filter/(2*self.numOfTestTriple)
        
        return Hits10_Raw,Hits10_filter,mean_Rank_Raw,mean_Rank_filter
        
    def ILP_relation(self):
        mean_Rank=0.0
        Hits1=0.0
        mean_Rank_f=0.0
        Hits1_f=0.0
        for i in range(self.numOfTestTriple):
            self.testHeadEmbedding = self.Transmit.entity_embeddings(self.testHead[i])
            self.testRelationEmbedding = self.Transmit.relation_embeddings(self.testRelation[i])
            self.testTailEmbedding = self.Transmit.entity_embeddings(self.testTail[i])
            targetLoss_trans = torch.norm(self.testHeadEmbedding + self.testRelationEmbedding - self.testTailEmbedding, self.norm).repeat(self.numOfRelation, 1)
            targetLoss_trans_tmp = torch.norm(self.testHeadEmbedding + self.testRelationEmbedding - self.testTailEmbedding, self.norm)
            tmpHeadEmbedding = self.testHeadEmbedding.repeat(self.numOfRelation, 1)
            tmpTailEmbedding = self.testTailEmbedding.repeat(self.numOfRelation, 1)
            
                
            tmpRelationLoss_trans_tmp=torch.norm(tmpHeadEmbedding+self.Transmit.relation_embeddings.weight.data-tmpTailEmbedding,self.norm,1)
            tmpRelationLoss_trans=torch.norm(tmpHeadEmbedding+self.Transmit.relation_embeddings.weight.data-tmpTailEmbedding,self.norm,1).view(-1,1)#10*1
                
            pre_relations,aft_relations=self.find_relations(self.testHead[i],self.testTime[i],self.seq_withTime)
            pre_relation_embeddings=self.Transmit.relation_embeddings(pre_relations)
            aft_relation_embeddings=self.Transmit.relation_embeddings(aft_relations)
            trans=torch.mm(self.Transmit.relation_trans,torch.from_numpy(np.linalg.pinv(self.Transmit.relation_trans.numpy())))
            if(len(pre_relations)!=0):
                Pre_RelationEmbedding = self.testRelationEmbedding.repeat(len(pre_relations),1)
                targetLoss_Transmit_pre=torch.sum(torch.norm(torch.mm(pre_relation_embeddings,trans)-Pre_RelationEmbedding,self.norm,1))
            else:
                targetLoss_Transmit_pre=0
            if(len(aft_relations)!=0):
                aft_RelationEmbedding = self.testRelationEmbedding.repeat(len(aft_relations),1)
                targetLoss_Transmit_aft=torch.sum(torch.norm(torch.mm(aft_RelationEmbedding,trans)-aft_relation_embeddings,self.norm,1))
            else:
                targetLoss_Transmit_aft=0
                    
            if(len(pre_relations)==0 and len(aft_relations)==0): 
                targetLoss_Transmit = 0
            else:
                targetLoss_Transmit = (targetLoss_Transmit_aft+targetLoss_Transmit_pre)/(len(pre_relations)+len(aft_relations))
            
            C_1=self.overlap_constrain(int(self.testHead[i]),int(self.testRelation[i]),int(self.testTail[i]),self.testTime[i].numpy())
            C_2_pre=self.order_constrain(int(self.testHead[i]),int(self.testRelation[i]),int(self.testTail[i]),self.testTime[i].numpy(),"pre")
            C_2_aft=self.order_constrain(int(self.testHead[i]),int(self.testRelation[i]),int(self.testTail[i]),self.testTime[i].numpy(),"aft")
                

            if C_1*C_2_pre*C_2_aft ==0 or (0.1*targetLoss_Transmit+float(targetLoss_trans_tmp)) > 13:
                targetLoss_Transmit=torch.FloatTensor([1000])
                
            targetLoss_Transmit = torch.FloatTensor([targetLoss_Transmit]).repeat(self.numOfRelation,1)
            
            loss=[]
                
            for j in range(self.numOfRelation):
                if(len(pre_relations)!=0):
                    Pre_RelationEmbedding=self.Transmit.relation_embeddings(torch.tensor(j)).repeat(len(pre_relations),1)
                    tmpLoss_Transmit_pre=torch.sum(torch.norm(torch.mm(pre_relation_embeddings,trans)-Pre_RelationEmbedding,self.norm,1))
                else:
                    tmpLoss_Transmit_pre=0
                if(len(aft_relations)!=0):
                    aft_RelationEmbedding=self.Transmit.relation_embeddings(torch.tensor(j)).repeat(len(aft_relations),1)
                    tmpLoss_Transmit_aft=torch.sum(torch.norm(torch.mm(aft_RelationEmbedding,trans)-aft_relation_embeddings,self.norm,1))
                else:
                    tmpLoss_Transmit_aft=0
                if(len(pre_relations)==0 and len(aft_relations)==0):
                    tmpLoss_Transmit=0
                else:
                    tmpLoss_Transmit = (tmpLoss_Transmit_aft+tmpLoss_Transmit_pre)/(len(pre_relations)+len(aft_relations))
                    
                C_1=self.overlap_constrain(int(self.testHead[i]),int(j),int(self.testTail[i]),self.testTime[i].numpy())
                C_2_pre=self.order_constrain(int(self.testHead[i]),int(j),int(self.testTail[i]),self.testTime[i].numpy(),"pre")
                C_2_aft=self.order_constrain(int(self.testHead[i]),int(j),int(self.testTail[i]),self.testTime[i].numpy(),"aft")
                    

                if C_1*C_2_pre*C_2_aft ==0 or (0.1*tmpLoss_Transmit+float(tmpRelationLoss_trans_tmp[j])) > 13:
                    tmpLoss_Transmit=torch.FloatTensor([10000])
                        
            tmpLoss_Transmit = torch.FloatTensor([tmpLoss_Transmit])
            tmpLoss_Transmit=torch.unsqueeze(tmpLoss_Transmit,0)
            tmpLoss_Transmit=torch.unsqueeze(tmpLoss_Transmit,1)
            loss.append(tmpLoss_Transmit)
             
            tmpRelationLoss_Transmit=torch.cat(loss,0)

            wrongRelation = torch.nonzero(nn.functional.relu(0.1*targetLoss_Transmit+targetLoss_trans-tmpRelationLoss_trans-0.1*tmpRelationLoss_Transmit))
            rankR=torch.nonzero(nn.functional.relu(0.1*targetLoss_Transmit+targetLoss_trans-tmpRelationLoss_trans-0.1*tmpRelationLoss_Transmit)).size()[0]+1

            numOfFilterRelation=0
            for tmpWrongRelation in wrongRelation:

                numOfFilterRelation += self.trainTriple_filter[(self.trainTriple_filter[:,0]==self.testHead[i].float())&(self.trainTriple_filter[:,1]==tmpWrongRelation[0].float())&(self.trainTriple_filter[:,2]==self.testTail[i].float())].size()[0]
                
            rankR_f=rankR-numOfFilterRelation
            mean_Rank_f=mean_Rank_f+rankR_f
            mean_Rank=mean_Rank+rankR
            if rankR==1:
                Hits1=Hits1+1
            if rankR_f==1:
                Hits1_f=Hits1_f+1
    
        mean_Rank=mean_Rank/self.numOfTestTriple
        Hits1=Hits1/self.numOfTestTriple
        mean_Rank_f=mean_Rank_f/self.numOfTestTriple
        Hits1_f=Hits1_f/self.numOfTestTriple
             
        return mean_Rank,Hits1,mean_Rank_f,Hits1_f  
    
    def ILP_solver(self):
        print ("-----Entity ILP Started at " + time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())) + "-----")
        
        Hits10_Raw,Hits10_filter,meanRank_Raw,meanRank_filter=self.ILP_entity()
        
        print ("-----Result of Entity Prediction (Raw)-----")
        print ("|  Mean Rank Raw |    |")
        print ("|  " + str(meanRank_Raw) + "  |  under implementing  |")
        print ("-----Result of Link Prediction (Filter)-----")
        print ("|  Mean Rank filter  |   |")
        print ("|  " + str(meanRank_filter) + "  |  under implementing  |")
        print ("|  Hits 10 Raw |   |")
        print ("|  " + str(Hits10_Raw) + "  |  under implementing  |")
        print ("-----Result of Link Prediction (Filter)-----")
        print ("|  Hits 10 filter  |    |")
        print ("|  " + str(Hits10_filter) + "  |  under implementing  |")
        
        print ("-----Fast Test Ended at " + time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())) + "-----")
        
        print ("-----Relation Test Started at " + time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())) + "-----")
        
        meanRank,Hits1,meanRank_f,Hits1_f=self.ILP_relation()
        
        print ("-----Result of Relation Prediction (Raw)-----")
        print ("|  Mean Rank  |    |")
        print ("|  " + str(meanRank) + "  |  under implementing  |")
        print ("|  Hits 1   |    |")
        print ("|  " + str(Hits1) + "  |  under implementing  |")
        print ("|  Mean Rank_F  |    |")
        print ("|  " + str(meanRank_f) + "  |  under implementing  |")
        print ("|  Hits 1_F   |    |")
        print ("|  " + str(Hits1_f) + "  |  under implementing  |")
        
        print ("-----Fast Test Ended at " + time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())) + "-----")
    
    def find_relations(self,head,time,seq):
        List=seq[int(head)]
        pre_rs=[]
        aft_rs=[]
        if(len(List)!=0):
            for i in List:
                if(i[0]<int(time[0])):
                    pre_rs.append(int(i[1]))
                if(i[0]>int(time[0])):
                    aft_rs.append(int(i[1]))
            
        return torch.LongTensor(pre_rs),torch.LongTensor(aft_rs)
    
    def preRead(self):
        #print "-----Reading Pre-Trained Results from " + self.preAdd + "-----"
        modelInput = open("./dataset/"+self.dataset + "/model.pickle", "rb")
        self.Transmit = pickle.load(modelInput)
        modelInput.close()
    
    def write(self,train_triples,path):
        train_ILP= "./dataset/"+self.dataset+"/"+path+".pickle"
        Output = open(train_ILP, "wb")
        pickle.dump(train_triples, Output)
        Output.close()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ILP")
    parser.add_argument("--dataset",dest="dataset",type=str,default="YAGO11K")
    #[YAGO11K,WIKI12K,WIKI36K]
    
    args=parser.parse_args()
    ILP(args)