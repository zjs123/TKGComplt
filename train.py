# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 17:04:27 2020

@author: zjs
"""

import re
import pickle
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Transmit, TKGFrame
from readTrainingData import readData
from generatePosAndCorBatch import generateBatches, dataset
import random
import Loss
import numpy as np
from ILP import ILP

class Train:
    
    def __init__(self,args):
        self.entityDimension = args.dimension
        self.relationDimension = args.dimension
        self.numOfEpochs = args.numOfEpochs
        self.numOfBatches = args.numOfBatches
        self.learningRate = args.lr
        self.margin_triple = args.margin
        self.margin_relation = args.margin
        self.trade_off = args.tf
        self.norm = args.norm
        self.dataset = args.dataset
        
        self.Triples = None
        self.train2id = {}
        self.seq_withTime = {}
        self.seq_relation = {}
        self.headRelation2Tail = {}
        self.tailRelation2Head = {}
        self.headTail2Relation = {}
        self.positiveBatch = {}
        self.corruptedBatch = {}
        self.relation_pair_batch = {}
        
        self.nums = [0,0,0]
        self.numOfTrainTriple = 0
        self.numOfEntity = 0
        self.numOfRelation = 0
        
        self.validate2id = {}
        self.validateHead = None
        self.validateRelation = None
        self.validateTail = None
        self.validateTime = None
        self.numOfValidateTriple = 0

        self.test2id = {}
        self.testHead = None
        self.testRelation = None
        self.testTail = None
        self.testTime = None
        self.numOfTestTriple = 0
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
            
        self.setup_seed(1)
        
        self.Transmit = None

        self.train()
        
        self.write()
        
        self.test_relation()
        
        self.test_entity()
    
    
    def setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        
    def train(self):
        path = "./dataset/"+self.dataset
        data=readData(path, self.train2id, self.seq_withTime, self.seq_relation,self.headRelation2Tail,self.tailRelation2Head,
                        self.headTail2Relation,self.nums)
        
        self.Triples = data.out()
        
        self.numOfTrainTriple = self.nums[0]
        self.numOfEntity = self.nums[1]
        self.numOfRelation = self.nums[2]
        
        self.readValidateTriples(path)
        self.readTestTriples(path)
        
        #self.Transmit = Transmit(self.numOfEntity, self.numOfRelation, self.entityDimension, self.relationDimension,
        #              self.norm)
        
        self.Transmit = TKGFrame(self.numOfEntity, self.numOfRelation, self.entityDimension, self.relationDimension,
                      self.norm)
        
        self.Transmit.to(self.device)
        
        
        
        
        #self.perRead(self.Transmit)
        
        Margin_Loss_H = Loss.marginLoss()
        Margin_Loss_D = Loss.double_marginLoss()
        
        optimizer = optim.SGD(self.Transmit.parameters(),lr=self.learningRate)
        
        Dataset = dataset(self.numOfTrainTriple)
        batchsize = int(self.numOfTrainTriple / self.numOfBatches)
        dataLoader = DataLoader(Dataset, batchsize, True)
        
        #self.write()
        for epoch in range(self.numOfEpochs):
            epochLoss = 0
            for batch in dataLoader:
                self.positiveBatch = {}
                self.corruptedBatch = {}
                self.relation_pair_batch = {}
                generateBatches(batch, self.train2id, self.seq_relation, self.positiveBatch, self.corruptedBatch, self.relation_pair_batch,self.numOfEntity,
                                self.numOfRelation,self.headRelation2Tail, self.tailRelation2Head, self.headTail2Relation)
                optimizer.zero_grad()
                positiveBatchHead = self.positiveBatch["h"].to(self.device)
                positiveBatchRelation = self.positiveBatch["r"].to(self.device)
                positiveBatchTail = self.positiveBatch["t"].to(self.device)
                corruptedBatchHead = self.corruptedBatch["h"].to(self.device)
                corruptedBatchRelation = self.corruptedBatch["r"].to(self.device)
                corruptedBatchTail = self.corruptedBatch["t"].to(self.device)
                relation_pair_h = self.relation_pair_batch["h"].to(self.device)
                relation_pair_t = self.relation_pair_batch["t"].to(self.device)
                relation_pair_step = self.relation_pair_batch["step"].to(self.device)
                positiveLoss,negativeLoss,positive_relation_pair_Loss,negative_relation_pair_Loss =self.Transmit(positiveBatchHead, positiveBatchRelation, positiveBatchTail,corruptedBatchHead,
                                   corruptedBatchRelation, corruptedBatchTail,relation_pair_h,relation_pair_t,relation_pair_step)
                transLoss = Margin_Loss_H(positiveLoss, negativeLoss, self.margin_triple)
                
                relationLoss = Margin_Loss_D(positive_relation_pair_Loss, negative_relation_pair_Loss, self.margin_relation)
                
                ent_embeddings = self.Transmit.entity_embeddings(torch.cat([positiveBatchHead, positiveBatchTail, corruptedBatchHead, corruptedBatchTail]))
                rel_embeddings = self.Transmit.relation_embeddings(torch.cat([positiveBatchRelation,relation_pair_h,relation_pair_t]))
            
                normloss=Loss.normLoss(ent_embeddings)+Loss.normLoss(rel_embeddings)#+Loss.F_norm(self.Transmit.relation_trans)#+0.1*Loss.orthogonalLoss(rel_embeddings,self.Transmit.relation_trans)
                batchLoss=transLoss+normloss+self.trade_off*relationLoss
                batchLoss.backward()
                optimizer.step()
                epochLoss += batchLoss
                
            print ("epoch " + str(epoch) + ": , loss: " + str(epochLoss))
            
            #ValidMR_entity = self.validate_entity()
            
            ValidMR_relation = self.validate_relation()
            
            #print("valid entity MR: "+str(ValidMR_entity))
            print("valid relation MR: "+str(ValidMR_relation))
            
            
    def validate_entity(self):
        meanRank = 0
        for tmpTriple in range(self.numOfValidateTriple):
            meanRank += self.Transmit.fastValidate_entity(self.validateHead[tmpTriple], self.validateRelation[tmpTriple], self.validateTail[tmpTriple])
        return meanRank/self.numOfValidateTriple
    
    def validate_relation(self):
        meanRank = 0
        for tmpTriple in range(self.numOfValidateTriple):
            meanRank += self.Transmit.fastValidate_relation(self.validateHead[tmpTriple], self.validateRelation[tmpTriple], self.validateTail[tmpTriple],self.validateTime[tmpTriple])
        return meanRank/self.numOfValidateTriple
    
    def test_relation(self):
        meanRank, Hits1, meanRank_F, Hits1_F= self.Transmit.test_relation(self.testHead, self.testRelation, self.testTail,self.Triples,self.numOfTestTriple)
        print("test_mean_rank_R: "+str(meanRank))
        print("test_Hits_1_R: "+str(Hits1))
        print("test_mean_rank_FR: "+str(meanRank_F))
        print("test_Hits_1_FR: "+str(Hits1_F))
        
    def test_entity(self):
        meanRank, Hits10, meanRank_F, Hits10_F= self.Transmit.test_entity(self.testHead, self.testRelation, self.testTail,self.Triples,self.numOfTestTriple)
        print("test_mean_rank_E: "+str(meanRank))
        print("test_Hits_10_E: "+str(Hits10))
        print("test_mean_rank_R: "+str(meanRank_F))
        print("test_Hits_1_R: "+str(Hits10_F))
            

    def write(self):
        #print "-----Writing Training Results to " + self.outAdd + "-----"
        transmit_path = "./dataset/"+self.dataset + "/model.pickle"
        modelOutput = open(transmit_path, "wb")
        pickle.dump(self.Transmit, modelOutput)
        modelOutput.close()

    def preRead(self):
        #print "-----Reading Pre-Trained Results from " + self.preAdd + "-----"
        modelInput = open("./dataset/"+self.dataset + "/model.pickle", "rb")
        self.Transmit = pickle.load(modelInput)
        modelInput.close()
        
        

    def readValidateTriples(self,path):
        fileName = path+"/valid.txt"
        print ("-----Reading Validation Triples from " +fileName + "-----")
        count = 0
        self.validate2id["h"] = []
        self.validate2id["r"] = []
        self.validate2id["t"] = []
        self.validate2id["time"] = []
        inputData = open(fileName)
        line = inputData.readline()
        self.numOfValidateTriple = int(re.findall(r"\d+", line)[0])
        lines = inputData.readlines()
        count=0
        for line in lines:
            count=count+1
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
            self.validate2id["h"].append(head)
            self.validate2id["r"].append(relation)
            self.validate2id["t"].append(tail)
            self.validate2id["time"].append([time_start,time_end])
        inputData.close()
        
        if count == self.numOfValidateTriple:
            print ("number of validation triples: " + str(self.numOfValidateTriple))
            self.validateHead = torch.LongTensor(self.validate2id["h"])
            self.validateRelation = torch.LongTensor(self.validate2id["r"])
            self.validateTail = torch.LongTensor(self.validate2id["t"])
            self.validateTime = torch.LongTensor(self.validate2id["time"])
        else:
            print ("error in " + fileName)
            
    def readTestTriples(self,path):
        fileName =path+"/test.txt"
        print ("-----Reading Test Triples from " + fileName + "-----")
        count = 0
        self.test2id["h"] = []
        self.test2id["r"] = []
        self.test2id["t"] = []
        self.test2id["time"] = []
        inputData = open( fileName)
        line = inputData.readline()
        self.numOfTestTriple = int(re.findall(r"\d+", line)[0])
        lines = inputData.readlines()
        count=0
        for line in lines:
            count=count+1
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
            self.test2id["h"].append(head)
            self.test2id["r"].append(relation)
            self.test2id["t"].append(tail)
            self.test2id["time"].append([time_start,time_end])
        inputData.close()
        
        if count == self.numOfTestTriple:
            print ("number of test triples: " + str(self.numOfTestTriple))
            self.testHead = torch.LongTensor(self.test2id["h"])
            self.testRelation = torch.LongTensor(self.test2id["r"])
            self.testTail = torch.LongTensor(self.test2id["t"])
            self.testTime=torch.LongTensor(self.test2id["time"])
        else:
            print ("error in " + fileName)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transmit")
    parser.add_argument("--hidden",dest="dimension",type=int,default=100)
    parser.add_argument("--nbatch",dest="numOfBatches",type=int,default=100)
    parser.add_argument("--nepoch",dest="numOfEpochs",type=int,default=400)
    parser.add_argument("--margin",dest="margin",type=float,default=4.0)
    parser.add_argument("--dataset",dest="dataset",type=str,default="WIKI12K")
    #[YAGO11K,WIKI12K,WIKI11K]
    parser.add_argument("--lr",dest="lr",type=float,default=0.001)
    parser.add_argument("--tf",dest="tf",type=float,default=0.001)
    parser.add_argument("--norm",dest="norm",type=int,default=1)
    
    args=parser.parse_args()
    Train(args)