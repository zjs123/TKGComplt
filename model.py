# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 17:03:24 2020

@author: zjs
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Transmit(nn.Module):
    
    def __init__(self, numOfEntity, numOfRelation, entityDimension, relationDimension, norm):
        super(Transmit,self).__init__()

        self.numOfEntity = numOfEntity
        self.numOfRelation = numOfRelation
        self.entityDimension = entityDimension
        self.relationDimension = relationDimension
        self.norm = norm

        sqrtR = relationDimension**0.5
        sqrtE = entityDimension**0.5

        self.relation_embeddings = nn.Embedding(self.numOfRelation, self.relationDimension)
        self.relation_embeddings.weight.data = torch.FloatTensor(self.numOfRelation, self.relationDimension).uniform_(-6./sqrtR, 6./sqrtR)
        self.relation_embeddings.weight.data = F.normalize(self.relation_embeddings.weight.data, 2, 1)

        self.entity_embeddings = nn.Embedding(self.numOfEntity, self.entityDimension)
        self.entity_embeddings.weight.data = torch.FloatTensor(self.numOfEntity, self.entityDimension).uniform_(-6./sqrtE, 6./sqrtE)
        self.entity_embeddings.weight.data = F.normalize(self.entity_embeddings.weight.data, 2, 1)

        self.relation_trans = torch.FloatTensor(self.relationDimension,self.relationDimension).uniform_(-6./sqrtE, 6./sqrtE)
        self.relation_trans = F.normalize(self.relation_trans,2,1)

    def forward(self, positiveBatchHead, positiveBatchRelation, positiveBatchTail, corruptedBatchHead, corruptedBatchRelation, corruptedBatchTail ,relation_pair_h ,relation_pair_t):

        pH_embeddings = self.entity_embeddings(positiveBatchHead)
        pR_embeddings = self.relation_embeddings(positiveBatchRelation)
        pT_embeddings = self.entity_embeddings(positiveBatchTail)

        nH_embeddings = self.entity_embeddings(corruptedBatchHead)
        nR_embeddings = self.relation_embeddings(corruptedBatchRelation)
        nT_embeddings = self.entity_embeddings(corruptedBatchTail)

        R_Ph_embeddings = self.relation_embeddings(relation_pair_h)
        R_Pt_embeddings = self.relation_embeddings(relation_pair_t)

        R_Ph_embeddings=F.normalize(R_Ph_embeddings,2,1)
        R_Pt_embeddings=F.normalize(R_Pt_embeddings,2,1)

        pH_embeddings = F.normalize(pH_embeddings, 2, 1)
        pT_embeddings = F.normalize(pT_embeddings, 2, 1)
        
        nH_embeddings = F.normalize(nH_embeddings, 2, 1)
        nT_embeddings = F.normalize(nT_embeddings, 2, 1)

        positiveLoss = torch.norm(pH_embeddings + pR_embeddings - pT_embeddings, self.norm, 1)
        negativeLoss = torch.norm(nH_embeddings + nR_embeddings - nT_embeddings, self.norm, 1)
        
        trans=torch.mm(self.relation_trans,torch.from_numpy(np.linalg.pinv(self.relation_trans.numpy())))

        positive_relation_pair_Loss = torch.norm(torch.mm(R_Ph_embeddings,trans)-R_Pt_embeddings,self.norm,1)
        negative_relation_pair_Loss = torch.norm(torch.mm(R_Pt_embeddings,trans)-R_Ph_embeddings,self.norm,1)
        

        return positiveLoss,negativeLoss,positive_relation_pair_Loss,negative_relation_pair_Loss
    
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
    
    def fastValidate_entity(self, validateHead, validateRelation, validateTail):
        validateHeadEmbedding = self.entity_embeddings(validateHead)
        validateRelationEmbedding = self.relation_embeddings(validateRelation)
        validateTailEmbedding = self.entity_embeddings(validateTail)
        targetLoss = torch.norm(validateHeadEmbedding + validateRelationEmbedding - validateTailEmbedding, self.norm).repeat(self.numOfEntity, 1)
        tmpHeadEmbedding = validateHeadEmbedding.repeat(self.numOfEntity, 1)
        tmpRelationEmbedding = validateRelationEmbedding.repeat(self.numOfEntity, 1)
        tmpTailEmbedding = validateTailEmbedding.repeat(self.numOfEntity, 1)

        tmpHeadLoss = torch.norm(self.entity_embeddings.weight.data + tmpRelationEmbedding - tmpTailEmbedding,
                                 self.norm, 1).view(-1, 1)
        tmpTailLoss = torch.norm(tmpHeadEmbedding + tmpRelationEmbedding - self.entity_embeddings.weight.data,
                                 self.norm, 1).view(-1, 1)

        rankH = torch.nonzero(nn.functional.relu(targetLoss - tmpHeadLoss)).size()[0]+1
        rankT = torch.nonzero(nn.functional.relu(targetLoss - tmpTailLoss)).size()[0]+1

        return (rankH + rankT + 2)/2
    
    def fastValidate_relation(self,validateHead, validateRelation, validateTail,validateTime,seq_withtime):
         validateHeadEmbedding = self.entity_embeddings(validateHead)
         validateRelationEmbedding = self.relation_embeddings(validateRelation)
         validateTailEmbedding = self.entity_embeddings(validateTail)
         targetLoss_trans = torch.norm(validateHeadEmbedding + validateRelationEmbedding - validateTailEmbedding, self.norm).repeat(self.numOfRelation, 1)
         tmpHeadEmbedding = validateHeadEmbedding.repeat(self.numOfRelation, 1)
         tmpTailEmbedding = validateTailEmbedding.repeat(self.numOfRelation, 1)
            
         tmpRelationLoss_trans=torch.norm(tmpHeadEmbedding+self.relation_embeddings.weight.data-tmpTailEmbedding,self.norm,1).view(-1,1)
            
         pre_relations,aft_relations=self.find_relations(validateHead,validateTime,seq_withtime)
         pre_relation_embeddings=self.relation_embeddings(pre_relations)
         aft_relation_embeddings=self.relation_embeddings(aft_relations)
            
         trans=torch.mm(self.relation_trans,torch.from_numpy(np.linalg.pinv(self.relation_trans.numpy())))
         if(len(pre_relations)!=0):
             Pre_RelationEmbedding = validateRelationEmbedding.repeat(len(pre_relations),1)
             targetLoss_Transmit_pre=torch.sum(torch.norm(torch.mm(pre_relation_embeddings,trans)-Pre_RelationEmbedding,self.norm,1))
         else:
             targetLoss_Transmit_pre=0
         if(len(aft_relations)!=0):
             aft_RelationEmbedding = validateRelationEmbedding.repeat(len(aft_relations),1)
             targetLoss_Transmit_aft=torch.sum(torch.norm(torch.mm(aft_RelationEmbedding,trans)-aft_relation_embeddings,self.norm,1))
         else:
             targetLoss_Transmit_aft=0
         if(len(pre_relations)!=0 and len(aft_relations)!=0): 
             targetLoss_Transmit = (targetLoss_Transmit_aft+targetLoss_Transmit_pre)/(len(pre_relations)+len(aft_relations))
         else:
             targetLoss_Transmit = torch.FloatTensor(0)
         targetLoss_Transmit = targetLoss_Transmit.repeat(self.numOfRelation,1)
            
         loss=[]
         for i in range(self.numOfRelation):
             if(len(pre_relations)!=0):
                 Pre_RelationEmbedding=self.relation_embeddings(torch.tensor(i)).repeat(len(pre_relations),1)
                 tmpLoss_Transmit_pre=torch.sum(torch.norm(torch.mm(pre_relation_embeddings,trans)-Pre_RelationEmbedding,self.norm,1))
             else:
                 tmpLoss_Transmit_pre=0
             if(len(aft_relations)!=0):
                 aft_RelationEmbedding=self.relation_embeddings(torch.tensor(i)).repeat(len(aft_relations),1)
                 tmpLoss_Transmit_aft=torch.sum(torch.norm(torch.mm(aft_RelationEmbedding,trans)-aft_relation_embeddings,self.norm,1))
             else:
                 tmpLoss_Transmit_aft=0
             if(len(pre_relations)!=0 and len(aft_relations)!=0): 
                 tmpLoss_Transmit = (tmpLoss_Transmit_aft+tmpLoss_Transmit_pre)/(len(pre_relations)+len(aft_relations))
             else:
                 tmpLoss_Transmit=torch.FloatTensor(0)
                 
             tmpLoss_Transmit=torch.unsqueeze(tmpLoss_Transmit,0)
             tmpLoss_Transmit=torch.unsqueeze(tmpLoss_Transmit,1)
             loss.append(tmpLoss_Transmit)
             
         tmpRelationLoss_Transmit=torch.cat(loss,0)
         rankR=torch.nonzero(nn.functional.relu(targetLoss_Transmit+targetLoss_trans-tmpRelationLoss_trans-tmpRelationLoss_Transmit)).size()[0]+1
            
         return rankR
    