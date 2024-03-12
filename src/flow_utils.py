import torch
import numpy as np
import cv2
import torch.nn.functional as F
from src.utils import *
import sys
sys.path.append("./src/ebsynth/deps/gmflow/")
from gmflow.geometry import flow_warp

"""
==========================================================================
* warp_tensor(): warp and fuse tensors based on optical flow and mask
* get_single_mapping_ind(): get pixel index correspondence between two frames
* get_mapping_ind(): get pixel index correspondence between consecutive frames within a batch
==========================================================================
"""

@torch.no_grad()
def warp_tensor(sample, flows, occs, saliency, unet_chunk_size):
    """
    Warp images or features based on optical flow
    Fuse the warped imges or features based on occusion masks and saliency map
    """
    scale = sample.shape[2] * 1.0 / flows[0].shape[2]
    kernel = int(1 / scale)
    bwd_flow_ = F.interpolate(flows[1] * scale, scale_factor=scale, mode='bilinear')
    bwd_occ_ = F.max_pool2d(occs[1].unsqueeze(1), kernel_size=kernel) # (N-1)*1*H1*W1
    if scale == 1:
        bwd_occ_ = Dilate(kernel_size=13, device=sample.device)(bwd_occ_)
    fwd_flow_ = F.interpolate(flows[0] * scale, scale_factor=scale, mode='bilinear')
    fwd_occ_ = F.max_pool2d(occs[0].unsqueeze(1), kernel_size=kernel) # (N-1)*1*H1*W1 
    if scale == 1:
        fwd_occ_ = Dilate(kernel_size=13, device=sample.device)(fwd_occ_)    
    scale2 = sample.shape[2] * 1.0 / saliency.shape[2]
    saliency = F.interpolate(saliency, scale_factor=scale2, mode='bilinear')
    latent = sample.to(torch.float32)
    video_length = sample.shape[0] // unet_chunk_size
    warp_saliency = flow_warp(saliency, bwd_flow_)
    warp_saliency_ = flow_warp(saliency[0:1], fwd_flow_[video_length-1:video_length])
    
    for j in range(unet_chunk_size):
        for ii in range(video_length-1):
            i = video_length * j + ii
            warped_image = flow_warp(latent[i:i+1], bwd_flow_[ii:ii+1])
            mask = (1 - bwd_occ_[ii:ii+1]) * saliency[ii+1:ii+2] * warp_saliency[ii:ii+1]
            latent[i+1:i+2] = latent[i+1:i+2] * (1-mask) + warped_image * mask
        i = video_length * j
        ii = video_length - 1
        warped_image = flow_warp(latent[i:i+1], fwd_flow_[ii:ii+1])
        mask = (1 - fwd_occ_[ii:ii+1]) * saliency[ii:ii+1] * warp_saliency_
        latent[ii+i:ii+i+1] = latent[ii+i:ii+i+1] * (1-mask) + warped_image * mask
        
    return latent.to(sample.dtype)


@torch.no_grad()
def get_single_mapping_ind(bwd_flow, bwd_occ, imgs, scale=1.0):
    """
    FLATTEN: Optical fLow-guided attention (Temoporal-guided attention)
    Find the correspondence between every pixels in a pair of frames
    
    [input]
    bwd_flow: 1*2*H*W   
    bwd_occ: 1*H*W      i.e., f2 = warp(f1, bwd_flow) * bwd_occ
    imgs: 2*3*H*W       i.e., [f1,f2]
    
    [output]
    mapping_ind: pixel index correspondence
    unlinkedmask: indicate whether a pixel has no correspondence
    i.e., f2 = f1[mapping_ind] * unlinkedmask
    """
    flows = F.interpolate(bwd_flow, scale_factor=1./scale, mode='bilinear')[0][[1,0]] / scale # 2*H*W
    _, H, W = flows.shape
    masks = torch.logical_not(F.interpolate(bwd_occ[None], scale_factor=1./scale, mode='bilinear') > 0.5)[0] # 1*H*W
    frames = F.interpolate(imgs, scale_factor=1./scale, mode='bilinear').view(2, 3, -1) # 2*3*HW
    grid = torch.stack(torch.meshgrid([torch.arange(H), torch.arange(W)]), dim=0).to(flows.device) # 2*H*W
    warp_grid = torch.round(grid + flows)
    mask = torch.logical_and(torch.logical_and(torch.logical_and(torch.logical_and(warp_grid[0] >= 0, warp_grid[0] < H),
                         warp_grid[1] >= 0), warp_grid[1] < W), masks[0]).view(-1) # HW
    warp_grid = warp_grid.view(2, -1) # 2*HW
    warp_ind = (warp_grid[0] * W + warp_grid[1]).to(torch.long)  # HW
    mapping_ind = torch.zeros_like(warp_ind) - 1 # HW
    
    for f0ind, f1ind in enumerate(warp_ind):
        if mask[f0ind]:
            if mapping_ind[f1ind] == -1:
                mapping_ind[f1ind] = f0ind
            else:
                targetv = frames[0,:,f1ind]
                pref0ind = mapping_ind[f1ind]
                prev = frames[1,:,pref0ind]
                v = frames[1,:,f0ind]
                if ((prev - targetv)**2).mean() > ((v - targetv)**2).mean():
                    mask[pref0ind] = False 
                    mapping_ind[f1ind] = f0ind
                else:
                    mask[f0ind] = False
                    
    unusedind = torch.arange(len(mask)).to(mask.device)[~mask]
    unlinkedmask = mapping_ind == -1
    mapping_ind[unlinkedmask] = unusedind
    return mapping_ind, unlinkedmask


@torch.no_grad()
def get_mapping_ind(bwd_flows, bwd_occs, imgs, scale=1.0):
    """
    FLATTEN: Optical fLow-guided attention (Temoporal-guided attention)
    Find pixel correspondence between every consecutive frames in a batch
    
    [input]
    bwd_flow: (N-1)*2*H*W   
    bwd_occ: (N-1)*H*W        
    imgs: N*3*H*W             
    
    [output]
    fwd_mappings: N*1*HW 
    bwd_mappings: N*1*HW 
    flattn_mask: HW*1*N*N
    i.e., imgs[i,:,fwd_mappings[i]] corresponds to imgs[0]
    i.e., imgs[i,:,fwd_mappings[i]][:,bwd_mappings[i]] restore the original imgs[i]
    """
    N, H, W = imgs.shape[0], int(imgs.shape[2] // scale), int(imgs.shape[3] // scale)
    iterattn_mask = torch.ones(H*W, N, N, dtype=torch.bool).to(imgs.device) 
    for i in range(len(imgs)-1):
        one_mask = torch.ones(N, N, dtype=torch.bool).to(imgs.device)
        one_mask[:i+1,i+1:] = False
        one_mask[i+1:,:i+1] = False
        mapping_ind, unlinkedmask = get_single_mapping_ind(bwd_flows[i:i+1], bwd_occs[i:i+1], imgs[i:i+2], scale)
        if i == 0:
            fwd_mapping = [torch.arange(len(mapping_ind)).to(mapping_ind.device)]
            bwd_mapping = [torch.arange(len(mapping_ind)).to(mapping_ind.device)]
        iterattn_mask[unlinkedmask[fwd_mapping[-1]]] = torch.logical_and(iterattn_mask[unlinkedmask[fwd_mapping[-1]]], one_mask)
        fwd_mapping += [mapping_ind[fwd_mapping[-1]]]
        bwd_mapping += [torch.sort(fwd_mapping[-1])[1]]
    fwd_mappings = torch.stack(fwd_mapping, dim=0).unsqueeze(1)
    bwd_mappings = torch.stack(bwd_mapping, dim=0).unsqueeze(1)
    return fwd_mappings, bwd_mappings, iterattn_mask.unsqueeze(1)

