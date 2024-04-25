import torch 
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np
import itertools

from backbone import *
from utils import *
from torchvision.ops import *

class Basenet(nn.Module):
    """
    main module of base model for collective dataset
    """
    def __init__(self, cfg):
        super(Basenet, self).__init__()
        self.cfg=cfg
        
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]
        NFB=self.cfg.num_features_boxes
        
        if cfg.backbone == 'resnet':
            self.backbone = MyResNet50(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'inception':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        
        # if not self.cfg.train_backbone:
        #     for p in self.backbone.parameters():
        #         p.requires_grad=False
        
        
        self.fc_emb_1=nn.Linear(K*K*D,NFB)
        self.dropout_emb_1 = nn.Dropout(p=self.cfg.train_dropout_prob)
        self.nl_emb_1=nn.LayerNorm([NFB])
        
        
        self.fc_actions=nn.Linear(NFB,self.cfg.num_actions)
        self.fc_activities_1=nn.Linear(2*NFB,NFB)
        self.fc_activities_2=nn.Linear(NFB,2)
        
        
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def savemodel(self,filepath):
        state = {
            'backbone_state_dict': self.backbone.state_dict(),
            'fc_emb_state_dict':self.fc_emb_1.state_dict(),
        }
        
        torch.save(state, filepath)
        print('model saved to:',filepath)
        

    def loadmodel(self,filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ',filepath)
        
                
    def forward(self,batch_data, seq_name, fid, normalized_bboxes):
        images_in, boxes_in, bboxes_num_in = batch_data
        
    
        # read config parameters
        B=images_in.shape[0]
        T=images_in.shape[1]
        H, W=self.cfg.image_size
        OH, OW=self.cfg.out_size
        MAX_N=self.cfg.num_boxes
        NFB=self.cfg.num_features_boxes
        
        D=self.cfg.emb_features # 1056
        K=self.cfg.crop_size[0]
        
        # Reshape the input data
        images_in_flat=torch.reshape(images_in,(B*T,3,H,W))  #B*T, 3, H, W
        boxes_in=boxes_in.reshape(B*T,MAX_N,4)
                
        # Use backbone to extract features of images_in
        # Pre-process first
        images_in_flat=prep_images(images_in_flat)
        outputs=self.backbone(images_in_flat)
        
            
        
        # Build multiscale features
        features_multiscale=[]
        for features in outputs:
            if features.shape[2:4]!=torch.Size([OH,OW]):
                features=F.interpolate(features,size=(OH,OW),mode='bilinear',align_corners=True)
            features_multiscale.append(features)
        
        features_multiscale=torch.cat(features_multiscale,dim=1)  #B*T, D, OH, OW
        # print(features_multiscale.size()) # torch.Size([B*T, 1056, 57, 87])  

        boxes_in_flat=torch.reshape(boxes_in,(B*T*MAX_N,4))  #B*T*MAX_N, 4
            
        boxes_idx=[i * torch.ones(MAX_N, dtype=torch.int)   for i in range(B*T) ]
        boxes_idx=torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, MAX_N
        boxes_idx_flat=torch.reshape(boxes_idx,(B*T*MAX_N,))  #B*T*MAX_N,

        boxes_idx_flat=boxes_idx_flat.float()
        boxes_idx_flat=torch.reshape(boxes_idx_flat,(-1,1))  #B*T*MAX_N,1

        # RoI Align
        boxes_in_flat.requires_grad=False
        boxes_idx_flat.requires_grad=False
       
        boxes_features_all=roi_align(features_multiscale,
                                            torch.cat((boxes_idx_flat,boxes_in_flat),1),
                                            (K,K)) 
        # print(boxes_features_all.shape) # torch.Size([20, 1056, 5, 5])
        
        boxes_features_all=boxes_features_all.reshape(B*T,MAX_N,-1)  #B*T,MAX_N, D*K*K torch.Size([5, 4, 26400])
        
        # Embedding 
        boxes_features_all=self.fc_emb_1(boxes_features_all)  # B*T,MAX_N, NFB
        boxes_features_all=self.nl_emb_1(boxes_features_all)
        boxes_features_all=F.relu(boxes_features_all)
        # boxes_features_all=self.dropout_emb_1(boxes_features_all)

        boxes_in=boxes_in.reshape(B,T,MAX_N,4)
        
    
        actions_scores=[]
        interaction_scores=[]

        bboxes_num_in=bboxes_num_in.reshape(B*T,)  #B*T,
        for bt in range(B*T):
            N=bboxes_num_in[bt]
            boxes_features=boxes_features_all[bt,:N,:].reshape(1,N,NFB)  #1,N,NFB
            boxes_states=boxes_features 
            
            # Predict actions
            boxes_states_flat=boxes_states.reshape(-1,NFB)  #1*N, NFS
            actn_score=self.fc_actions(boxes_states_flat)  #1*N, actn_num
            actions_scores.append(actn_score)

            # Predict activities
            interaction_flat=[]
            for i in range(N):
                for j in range(N):
                    if i!=j:  
                        interaction_flat.append(torch.cat([boxes_states_flat[i],boxes_states_flat[j]],dim=0))
            interaction_flat=torch.stack(interaction_flat,dim=0) #N(N-1),2048

            acty_score=self.fc_activities_1(interaction_flat)
            acty_score=F.relu(acty_score)  
            acty_score=self.fc_activities_2(acty_score) #1, acty_num
            interaction_scores.append(acty_score)
            
            
        actions_scores=torch.cat(actions_scores,dim=0)  #ALL_N,actn_num
        interaction_scores=torch.cat(interaction_scores,dim=0)
            
       
        return actions_scores, interaction_scores
        