import torch
from torch import nn
import torch.nn.modules.utils as nn_utils
import math
import timm
import types
import copy
import pandas
import numpy
import os
from typing import Union, List, Tuple
from timm.layers import SelectAdaptivePool2d, LayerNorm2d
import torch.nn.functional as F

class BinaryPooling2d(nn.Module):
    """
    Binary pooling layer as described in the algorithm.
    Parameters
    ----------
    kernel_size : int or tuple, default=3
        Size of the pooling window (must be square for now).
    stride : int, default=1
        Stride of the pooling operation.
    padding : int, default=0
        Optional zero padding.
    """
    def __init__(self, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        returns: (B, C, H_out, W_out)
        """
        B, C, H, W = x.shape
        k = self.kernel_size

        # unfold creates patches of size [B, C*k*k, L] where L = H_out * W_out
        patches = F.unfold(x, kernel_size=k, stride=self.stride, padding=self.padding)
        # reshape to [B, C, k*k, L]
        patches = patches.view(B, C, k * k, -1)

        # Step 4: center pixel = middle of flattened 3x3 window
        center_idx = (k * k) // 2
        center = patches[:, :, center_idx:center_idx + 1, :]   # shape [B,C,1,L]

        # Step 5: difference matrix
        diff = patches - center

        # Step 6: adaptive threshold = mean absolute diff
        threshold = diff.abs().mean(dim=2, keepdim=True)

        # Steps 7-13: binary mask
        binary_mask = (diff >= threshold).float()

        # Step 14: aggregated binary value
        binary_value = binary_mask.sum(dim=2, keepdim=True)

        # window mean
        window_mean = patches.mean(dim=2, keepdim=True)

        # Step 15: normalization factor
        norm_factor = (binary_value - window_mean) / 255.0

        # std and max over window
        window_std = patches.std(dim=2, keepdim=True, unbiased=False)
        window_max, _ = patches.max(dim=2, keepdim=True)

        # Step 16: normalized binary value
        binary_norm = norm_factor * window_std + (1 - norm_factor) * window_max

        # Step 17: assign to output
        # reshape back to (B, C, H_out, W_out)
        H_out = (H + 2 * self.padding - k) // self.stride + 1
        W_out = (W + 2 * self.padding - k) // self.stride + 1
        out = binary_norm.view(B, C, H_out, W_out)
        out = out.mean(dim=(-1, -2), keepdim=True)

        return out

class Extractor():

    def __init__(self, model: nn.Module = None, device: str = 'cuda', pooling_type='avg'):
        """
        :param model_type: A string specifying the type of model to extract from.
        :param stride: stride of first convolution layer. small stride -> higher resolution.
        :param model: Optional parameter. The nn.Module to extract from instead of creating a new one in ViTExtractor.
                      should be compatible with model_type.
        """
        model.head.fc = nn.Identity()
        
        # select model and device 
        self.device = device
        self.model = model

        # set model to eval mode and send to the device 
        self.model.eval()
        self.model.to(self.device)
        
        # set containers for results 
        self._feats = []
        self.hook_handlers = []
        self.load_size = None
        self.pooling_type = pooling_type
        self.num_patches = None
        self.norm_for_base = LayerNorm2d(320, eps=1e-5, affine=True)
        self.flatten_for_base = nn.Flatten(start_dim=1, end_dim=-1)
        if pooling_type == 'avg':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        else:
            self.pooling = BinaryPooling2d(kernel_size=3, stride=2) # well we can also try with sidderent kernel size and stride
            self.second_pooling = BinaryPooling2d(kernel_size=3, stride=1) 


    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """

        if facet in ['strategy_a','strategy_b']:
            def _hook(model, input, output):
                x = input[0]
                
                # get input value 
                x0 = self.pooling(x)
                x0_flattened = torch.flatten(x0, start_dim = 1, end_dim= -1)
                
                # perform all stage 3 operations 
                x = model.downsample(x)
                x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
                x_tmp = model.blocks[0](x)
                x_tmp2 = model.blocks[1](x_tmp)
                
                # permute 
                x_tmp = x_tmp.permute(0, 3, 1, 2)
                x_tmp2 = x_tmp2.permute(0, 3, 1, 2)
                x1 = self.pooling(x_tmp)
                x2 = self.pooling(x_tmp2)
                
                # flatten all values 
                x1_flattened = torch.flatten(x1, start_dim = 1, end_dim= -1)
                x2_flattened = torch.flatten(x2, start_dim = 1, end_dim= -1)
                
                # concatenate 
                #inter_out = torch.cat((x0_flattened,x1_flattened), dim=-1)
                #output = torch.cat((inter_out,x2_flattened), dim=-1)
                agg = torch.cat((x1_flattened,x2_flattened), dim=-1)
                #output = output.permute(0, 3, 1, 2)
                
                # append data to fets container 
                self._feats.append(agg)
            return _hook
            
        elif facet in ['strategy_c']:
            def _hook(model, input, output):
                x = input[0]
                
                # get input value 
                x0 = self.pooling(x)
                x0_flattened = torch.flatten(x0, start_dim = 1, end_dim= -1)
                
                # perform all stage 3 operations 
                x = model.downsample(x)
                x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
                x_tmp = model.blocks[0](x)
                x_tmp2 = model.blocks[1](x_tmp)
                
                # permute 
                x_tmp = x_tmp.permute(0, 3, 1, 2)
                x_tmp2 = x_tmp2.permute(0, 3, 1, 2)
                x1 = self.pooling(x_tmp)
                x2 = self.pooling(x_tmp2)
                
                # flatten all values 
                x1_flattened = torch.flatten(x1, start_dim = 1, end_dim= -1)
                x2_flattened = torch.flatten(x2, start_dim = 1, end_dim= -1)
                
                # concatenate 
                #inter_out = torch.cat((x0_flattened,x1_flattened), dim=-1)
                #output = torch.cat((inter_out,x2_flattened), dim=-1)
                agg = torch.cat((x0_flattened, x1_flattened,x2_flattened), dim=-1)
                #output = output.permute(0, 3, 1, 2)
                
                # append data to fets container 
                self._feats.append(agg)
            return _hook
            
        elif facet in ['strategy_d']:
            def _hook(model, input, output):
                # get ll stages and avg them 
                x = input[0]
                s0 = model[0](x)
                s1= model[1](s0)
                s2 = model[2](s1)
                s3= model[3](s2)
                
                # avg 
                s0_avg = self.pooling(s0)
                s1_avg = self.pooling(s1)
                s2_avg = self.pooling(s2)
                s3_avg = self.pooling(s3)
                
                # flatten
                s0_flat = torch.flatten(s0_avg, start_dim = 1, end_dim= -1)
                s1_flat = torch.flatten(s1_avg, start_dim = 1, end_dim= -1)
                s2_flat = torch.flatten(s2_avg, start_dim = 1, end_dim= -1)
                s3_flat = torch.flatten(s3_avg, start_dim = 1, end_dim= -1)
                
                # concat 
                output = torch.cat((s0_flat,s1_flat,s2_flat,s3_flat), dim=-1)
                
                # append results 
                self._feats.append(output)
            return _hook
                
        elif facet in ['base']:
            def _hook(model, input, output):
                # get ll stages and avg them 
                x = input[0]
                s0 = self.pooling(x)
                s1 = self.norm_for_base(s0)
                s2 = self.flatten_for_base(s1)
                
                # append results 
                self._feats.append(s2)
            return _hook
        
        else:
            raise TypeError(f"{facet} is not a supported facet.")


    def _register_hooks(self, mode: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        if mode: 
            print(mode)
            if mode in ['strategy_a','strategy_b','strategy_c']:
                facet = mode
                self.hook_handlers.append(self.model.stages[3].register_forward_hook(self._get_hook(facet)))
            elif mode in ['strategy_d']:
                facet = 'strategy_d'
                self.hook_handlers.append(self.model.stages.register_forward_hook(self._get_hook(facet)))
            elif mode in ['base']:
                if self.pooling_type == 'binary':
                    facet = mode
                    self.hook_handlers.append(self.model.head.register_forward_hook(self._get_hook(facet)))
                else:
                    pass 
            else: 
                raise TypeError(f"{mode} is not a supported extraction module.")

        else: 
            print('No mode provided, extracting according to provided layer and facet')
            print('Ughhh.. method not implemented yet...') 


    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(self, batch: torch.Tensor, mode :str = None) -> List[torch.Tensor]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """
        # get current batch shape 
        B, C, H, W = batch.shape
        
        # clear feats container and register current hook
        self._feats = []
        self._register_hooks(mode)

        
        # pass batch through the model and get last layer feats 
        llayer_feats = self.model(batch)
        self._unregister_hooks()
        if mode == "base" and self.pooling_type == 'avg':
            return llayer_feats
        elif mode in ['strategy_a']:
            return torch.cat((self._feats[0],llayer_feats), dim=-1)
        else:
            return self._feats


    def extract_descriptors(self, batch: torch.Tensor, mode: str = None) -> torch.Tensor:
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']
        :param bin: apply log binning to the descriptor. default is False.
        :return: tensor of descriptors. Bx1xtxd' where d' is the dimension of the descriptors.
        """

        feats = self._extract_features(batch, mode)
        if mode not in ['strategy_a','base']:
            x = feats[0]
        else: 
            x = feats 

        if mode == 'base' and  self.pooling_type == 'binary':
            x = feats[0]
        print(x.shape)

        return x
