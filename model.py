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
    
class TKGFrame(nn.Module):
    
    def __init__(self, numOfEntity, numOfRelation, entityDimension, relationDimension, norm):
        super(TKGFrame,self).__init__()

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

    def forward(self, positiveBatchHead, positiveBatchRelation, positiveBatchTail, corruptedBatchHead, corruptedBatchRelation, corruptedBatchTail ,relation_pair_h ,relation_pair_t, relation_pair_step):

        pH_embeddings = self.entity_embeddings(positiveBatchHead)
        pR_embeddings = self.relation_embeddings(positiveBatchRelation)
        pT_embeddings = self.entity_embeddings(positiveBatchTail)

        nH_embeddings = self.entity_embeddings(corruptedBatchHead)
        nR_embeddings = self.relation_embeddings(corruptedBatchRelation)
        nT_embeddings = self.entity_embeddings(corruptedBatchTail)


        pH_embeddings = F.normalize(pH_embeddings, 2, 1)
        pT_embeddings = F.normalize(pT_embeddings, 2, 1)
        
        nH_embeddings = F.normalize(nH_embeddings, 2, 1)
        nT_embeddings = F.normalize(nT_embeddings, 2, 1)

        positiveLoss = torch.norm(pH_embeddings + pR_embeddings - pT_embeddings, self.norm, 1)
        negativeLoss = torch.norm(nH_embeddings + nR_embeddings - nT_embeddings, self.norm, 1)
        
        #trans=torch.mm(self.relation_trans,torch.from_numpy(np.linalg.pinv(self.relation_trans.numpy())))
        

        #positive_relation_pair_Loss = torch.norm(torch.mm(R_Ph_embeddings,trans)-R_Pt_embeddings,self.norm,1)
        #negative_relation_pair_Loss = torch.norm(torch.mm(R_Pt_embeddings,trans)-R_Ph_embeddings,self.norm,1)
        
        positive_relation_pair_Loss,negative_relation_pair_Loss = self.caculate_relation_score(relation_pair_h, relation_pair_t, relation_pair_step)

        return positiveLoss,negativeLoss,positive_relation_pair_Loss,negative_relation_pair_Loss
    
    def caculate_relation_score(self,R_h_batch,R_t_batch,step_batch):
        R_h_embeddings = self.relation_embeddings(R_h_batch)
        R_t_embeddings = self.relation_embeddings(R_t_batch)

        R_h_embeddings=F.normalize(R_h_embeddings,2,1)
        R_t_embeddings=F.normalize(R_t_embeddings,2,1)
        
        R_h_embeddings_temp = []
        R_t_embeddings_temp = []
        for i in range(len(step_batch)):
            evolving_matrix = self.relation_trans.pow(float(step_batch[i]))
            #for k in range(step_batch[i]-1):
            #    evolving_matrix = torch.mm(evolving_matrix , self.relation_trans)
                
            R_h_embedding_sub = torch.matmul(R_h_embeddings[i],evolving_matrix).unsqueeze(0)
            R_t_embedding_sub = torch.matmul(R_t_embeddings[i],evolving_matrix).unsqueeze(0)
            
            R_h_embeddings_temp.append(R_h_embedding_sub)
            R_t_embeddings_temp.append(R_t_embedding_sub)
            
        R_h_embeddings_new=torch.cat(R_h_embeddings_temp,0)
        R_t_embeddings_new=torch.cat(R_t_embeddings_temp,0)
        
        positive_relation_pair_Loss = torch.norm(R_h_embeddings_new-R_t_embeddings,self.norm,1)
        negative_relation_pair_Loss = torch.norm(R_t_embeddings_new-R_h_embeddings,self.norm,1)
        
        return positive_relation_pair_Loss, negative_relation_pair_Loss
            
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
    
    def fastValidate_relation(self,validateHead, validateRelation, validateTail,validateTime):
        validateHeadEmbedding = self.entity_embeddings(validateHead)
        validateRelationEmbedding = self.relation_embeddings(validateRelation)
        validateTailEmbedding = self.entity_embeddings(validateTail)
        targetLoss = torch.norm(validateHeadEmbedding + validateRelationEmbedding - validateTailEmbedding, self.norm).repeat(self.numOfRelation, 1)
        tmpHeadEmbedding = validateHeadEmbedding.repeat(self.numOfRelation, 1)
        tmpTailEmbedding = validateTailEmbedding.repeat(self.numOfRelation, 1)

        tmpRelLoss = torch.norm(tmpHeadEmbedding + self.relation_embeddings.weight.data - tmpTailEmbedding,
                                 self.norm, 1).view(-1, 1)

        rankR = torch.nonzero(nn.functional.relu(targetLoss - tmpRelLoss)).size()[0]+1

        return rankR
    
    def test_relation(self,testHead,testRelation,testTail,trainTriple,numOfTestTriple):
        mean_Rank=0.0
        Hits1=0.0 
        mean_Rank_f=0.0
        Hits1_f=0.0
        
        for i in range(numOfTestTriple):
            testHeadEmbedding = self.entity_embeddings(testHead[i])
            testRelationEmbedding = self.relation_embeddings(testRelation[i])
            testTailEmbedding = self.entity_embeddings(testTail[i])
            targetLoss = torch.norm(testHeadEmbedding + testRelationEmbedding - testTailEmbedding, self.norm).repeat(self.numOfRelation, 1)
            tmpHeadEmbedding = testHeadEmbedding.repeat(self.numOfRelation, 1)
            #tmpRelationEmbedding = validateRelationEmbedding.repeat(self.numOfRelation, 1)
            tmpTailEmbedding = testTailEmbedding.repeat(self.numOfRelation, 1)
            
            tmpRelationLoss=torch.norm(tmpHeadEmbedding+self.relation_embeddings.weight.data-tmpTailEmbedding,self.norm,1).view(-1,1)
            rankR= torch.nonzero(nn.functional.relu(targetLoss - tmpRelationLoss)).size()[0]+1
            
            wrongRelation = torch.nonzero(nn.functional.relu(targetLoss - tmpRelationLoss))
                
            numOfFilterRelation=0
            for tmpWrongRelation in wrongRelation:
            #print(tmpWrongHead[0])
                numOfFilterRelation += trainTriple[(trainTriple[:,0]==testHead[i].float())&(trainTriple[:,1]==tmpWrongRelation[0].float())&(trainTriple[:,2]==testTail[i].float())].size()[0]
                
            rankR_f=rankR-numOfFilterRelation
            mean_Rank_f=mean_Rank_f+rankR_f
            mean_Rank=mean_Rank+rankR
            if rankR==1:
                Hits1=Hits1+1
            if rankR_f==1:
                Hits1_f=Hits1_f+1
        mean_Rank_f = mean_Rank_f/numOfTestTriple
        Hits1_f = Hits1_f/numOfTestTriple
        mean_Rank = mean_Rank/numOfTestTriple
        Hits1 = Hits1/numOfTestTriple
        
        return mean_Rank,Hits1,mean_Rank_f,Hits1_f
    
    def test_entity(self,testHead,testRelation,testTail,trainTriple,numOfTestTriple):
        mean_Rank_Raw=0.0
        mean_Rank_filter=0.0
        Hits10_Raw=0.0
        Hits10_filter=0.0
        for i in range(numOfTestTriple):
            testHeadEmbedding=self.entity_embeddings(testHead[i])
            testRelationEmbedding=self.relation_embeddings(testRelation[i])
            testTailEmbedding=self.entity_embeddings(testTail[i])
            targetLoss = torch.norm(testHeadEmbedding+testRelationEmbedding-testTailEmbedding,self.norm).repeat(self.numOfEntity,1)
            tmpHeadEmbedding=testHeadEmbedding.repeat(self.numOfEntity,1)
            tmpRelationEmbedding=testRelationEmbedding.repeat(self.numOfEntity,1)
            tmpTailEmbedding=testTailEmbedding.repeat(self.numOfEntity,1)
            
            tmpHeadLoss=torch.norm(self.entity_embeddings.weight.data + tmpRelationEmbedding - tmpTailEmbedding,
                                 self.norm, 1).view(-1, 1)
            tmpTailLoss = torch.norm(tmpHeadEmbedding + tmpRelationEmbedding - self.entity_embeddings.weight.data,
                                 self.norm, 1).view(-1, 1)
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
                #print(tmpWrongHead)
                numOfFilterHead += trainTriple[(trainTriple[:,0]==tmpWrongHead[0].float())&(trainTriple[:,1]==testRelation[i].float())&(trainTriple[:,2]==testTail[i].float())].size()[0]
            for tmpWrongTail in wrongTail:
                #print(tmpWrongTail)
                numOfFilterTail += trainTriple[(trainTriple[:,0]==testHead[i].float())&(trainTriple[:,1]==testRelation[i].float())&(trainTriple[:,2]==tmpWrongTail[0].float())].size()[0]
            
            Rank_H_filter=Rank_H-numOfFilterHead
            Rank_T_filter=Rank_T-numOfFilterTail
            
            mean_Rank_filter=mean_Rank_filter+Rank_H_filter+Rank_T_filter
            if Rank_H_filter<=10:
                Hits10_filter=Hits10_filter+1
            if Rank_T_filter<=10:
                Hits10_filter=Hits10_filter+1
        #print(len(wrongTail))
            
        Hits10_Raw=Hits10_Raw/(2*numOfTestTriple)
        Hits10_filter=Hits10_filter/(2*numOfTestTriple)
        mean_Rank_Raw=mean_Rank_Raw/(2*numOfTestTriple)
        mean_Rank_filter=mean_Rank_filter/(2*numOfTestTriple)
        
        return mean_Rank_Raw, Hits10_Raw, mean_Rank_filter, Hits10_filter