import torch
import random as rd
from torch.utils.data import Dataset


class dataset(Dataset):

    def __init__(self, numOfTriple):
        self.tripleList = torch.LongTensor(range(numOfTriple))
        self.numOfTriple = numOfTriple

    def __len__(self):
        return self.numOfTriple

    def __getitem__(self, item):
        return self.tripleList[item]


class generateBatches:

    def __init__(self, batch, train2id, seq_r, positiveBatch, corruptedBatch, relation_pair_Batch,numOfEntity, numOfRelation, headRelation2Tail, tailRelation2Head,headTail2Relation):
        self.batch = batch
        self.train2id = train2id
        self.seq_r=seq_r
        self.positiveBatch = positiveBatch
        self.corruptedBatch = corruptedBatch
        self.numOfEntity = numOfEntity
        self.numOfRelation = numOfRelation
        self.headRelation2Tail = headRelation2Tail
        self.tailRelation2Head = tailRelation2Head
        self.headTail2Relation = headTail2Relation
        self.relation_pair_Batch = relation_pair_Batch

        self.generatePosAndCorBatch()
        rd.seed(0.5)
        #self.generateRelationPairBatch()

    def generatePosAndCorBatch(self):
        self.positiveBatch["h"] = []
        self.positiveBatch["r"] = []
        self.positiveBatch["t"] = []
        self.corruptedBatch["h"] = []
        self.corruptedBatch["r"] = []
        self.corruptedBatch["t"] = []
        self.relation_pair_Batch["h"] = []
        self.relation_pair_Batch["t"] = []
        for tripleId in self.batch:
            tmpHead = self.train2id["h"][tripleId]
            tmpRelation = self.train2id["r"][tripleId]
            tmpTail = self.train2id["t"][tripleId]

            if(len(self.seq_r[tmpHead])>=2):
                pre=rd.randint(0,len(self.seq_r[tmpHead])-2)
                last=rd.randint(pre+1,len(self.seq_r[tmpHead])-1)
                self.relation_pair_Batch["h"].append(self.seq_r[tmpHead][pre])
                self.relation_pair_Batch["t"].append(self.seq_r[tmpHead][last])
            else:
                rand_index=rd.randint(0,self.numOfEntity-1)
                while len(self.seq_r[rand_index])<2:
                    #print(len(self.seq_r[rand_index]))
                    rand_index=rd.randint(0,self.numOfEntity-1)
                pre=rd.randint(0,len(self.seq_r[rand_index])-2)
                last=rd.randint(pre+1,len(self.seq_r[rand_index])-1)
                self.relation_pair_Batch["h"].append(self.seq_r[rand_index][pre])
                self.relation_pair_Batch["t"].append(self.seq_r[rand_index][last])
           
            #random=rd.random()
            for i in range(3):
                self.positiveBatch["h"].append(tmpHead)
                self.positiveBatch["r"].append(tmpRelation)
                self.positiveBatch["t"].append(tmpTail)
                random=torch.rand(1).item()
                if random <=0.5:
                    tmpCorruptedHead = torch.FloatTensor(1).uniform_(0, self.numOfEntity).long().item()
                    while tmpCorruptedHead in self.tailRelation2Head[tmpTail][tmpRelation] or tmpCorruptedHead == tmpHead:
                        tmpCorruptedHead = torch.FloatTensor(1).uniform_(0, self.numOfEntity).long().item()
                    self.corruptedBatch["h"].append(tmpCorruptedHead)
                    self.corruptedBatch["r"].append(tmpRelation)
                    self.corruptedBatch["t"].append(tmpTail)
                else:
                    tmpCorruptedTail = torch.FloatTensor(1).uniform_(0, self.numOfEntity).long().item()
                    #print(tmpCorruptedTail)
                    while tmpCorruptedTail in self.headRelation2Tail[tmpHead][tmpRelation] or tmpCorruptedTail == tmpTail:
                        tmpCorruptedTail = torch.FloatTensor(1).uniform_(0, self.numOfEntity).long().item()
                    self.corruptedBatch["h"].append(tmpHead)
                    self.corruptedBatch["r"].append(tmpRelation)
                    self.corruptedBatch["t"].append(tmpCorruptedTail)
        for aKey in self.positiveBatch:
            self.positiveBatch[aKey] = torch.LongTensor(self.positiveBatch[aKey])
        for aKey in self.corruptedBatch:
            self.corruptedBatch[aKey] = torch.LongTensor(self.corruptedBatch[aKey])
        for aKey in self.relation_pair_Batch:
            self.relation_pair_Batch[aKey] = torch.LongTensor(self.relation_pair_Batch[aKey])

