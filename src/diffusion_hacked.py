from einops import rearrange, reduce, repeat
import torch.nn.functional as F
import torch
import gc
from src.utils import *
from src.flow_utils import get_mapping_ind, warp_tensor
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.attention_processor import AttnProcessor2_0
from typing import Any, Dict, List, Optional, Tuple, Union
import sys
sys.path.append("./src/ebsynth/deps/gmflow/")
from gmflow.geometry import flow_warp, forward_backward_consistency_check

"""
==========================================================================
PART I - FRESCO-based attention
* Class AttentionControl: Control the function of FRESCO-based attention
* Class FRESCOAttnProcessor2_0: FRESCO-based attention
* apply_FRESCO_attn(): Apply FRESCO-based attention to a StableDiffusionPipeline
==========================================================================
"""

class AttentionControl():
    """
    Control FRESCO-based attention
    * enable/diable spatial-guided attention
    * enable/diable temporal-guided attention
    * enable/diable cross-frame attention
    * collect intermediate attention feature (for spatial-guided attention)
    """
    def __init__(self):
        self.stored_attn = self.get_empty_store()
        self.store = False
        self.index = 0
        self.attn_mask = None
        self.interattn_paras = None
        self.use_interattn = False
        self.use_cfattn = False
        self.use_intraattn = False
        self.intraattn_bias = 0
        self.intraattn_scale_factor = 0.2
        self.interattn_scale_factor = 0.2
    
    @staticmethod
    def get_empty_store():
        return {
            'decoder_attn': [],
        }
    
    def clear_store(self):
        del self.stored_attn
        torch.cuda.empty_cache()
        gc.collect()
        self.stored_attn = self.get_empty_store()
        self.disable_intraattn()

    # store attention feature of the input frame for spatial-guided attention
    def enable_store(self):
        self.store = True
        
    def disable_store(self):
        self.store = False  

    # spatial-guided attention
    def enable_intraattn(self):
        self.index = 0
        self.use_intraattn = True
        self.disable_store()
        if len(self.stored_attn['decoder_attn']) == 0:
            self.use_intraattn = False
        
    def disable_intraattn(self):
        self.index = 0
        self.use_intraattn = False
        self.disable_store()

    def disable_cfattn(self):
        self.use_cfattn = False        

    # cross frame attention
    def enable_cfattn(self, attn_mask=None):
        if attn_mask:
            if self.attn_mask:
                del self.attn_mask
                torch.cuda.empty_cache()
            self.attn_mask = attn_mask
            self.use_cfattn = True  
        else:
            if self.attn_mask:
                self.use_cfattn = True
            else:
                print('Warning: no valid cross-frame attention parameters available!')
                self.disable_cfattn()       
        
    def disable_interattn(self):
        self.use_interattn = False

    # temporal-guided attention
    def enable_interattn(self, interattn_paras=None):
        if interattn_paras:
            if self.interattn_paras:
                del self.interattn_paras
                torch.cuda.empty_cache()
            self.interattn_paras = interattn_paras
            self.use_interattn = True
        else:
            if self.interattn_paras:
                self.use_interattn = True
            else:
                print('Warning: no valid temporal-guided attention parameters available!')
                self.disable_interattn()
    
    def disable_controller(self):
        self.disable_intraattn()
        self.disable_interattn()
        self.disable_cfattn()
    
    def enable_controller(self, interattn_paras=None, attn_mask=None):
        self.enable_intraattn()
        self.enable_interattn(interattn_paras)
        self.enable_cfattn(attn_mask)    
    
    def forward(self, context):
        if self.store:
            self.stored_attn['decoder_attn'].append(context.detach())
        if self.use_intraattn and len(self.stored_attn['decoder_attn']) > 0:
            tmp = self.stored_attn['decoder_attn'][self.index]
            self.index = self.index + 1
            if self.index >= len(self.stored_attn['decoder_attn']):
                self.index = 0
                self.disable_store()
            return tmp
        return context
    
    def __call__(self, context):
        context = self.forward(context)
        return context


#import xformers
#import importlib
class FRESCOAttnProcessor2_0:
    """
    Hack self attention to FRESCO-based attention
    * adding spatial-guided attention
    * adding temporal-guided attention
    * adding cross-frame attention
    
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    Usage
    frescoProc = FRESCOAttnProcessor2_0(2, attn_mask)
    attnProc = AttnProcessor2_0()
    
    attn_processor_dict = {}
    for k in pipe.unet.attn_processors.keys():
        if k.startswith("up_blocks.2") or k.startswith("up_blocks.3"):
            attn_processor_dict[k] = frescoProc
        else:
            attn_processor_dict[k] = attnProc
    pipe.unet.set_attn_processor(attn_processor_dict)
    """

    def __init__(self, unet_chunk_size=2, controller=None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.unet_chunk_size = unet_chunk_size
        self.controller = controller
            
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        crossattn = False
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            if self.controller and self.controller.store:
                self.controller(hidden_states.detach().clone())
        else:
            crossattn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            
        # BC * HW * 8D
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        query_raw, key_raw = None, None
        if self.controller and self.controller.use_interattn and (not crossattn):
            query_raw, key_raw = query.clone(), key.clone()

        inner_dim = key.shape[-1] # 8D
        head_dim = inner_dim // attn.heads # D
        
        '''for efficient cross-frame attention'''
        if self.controller and self.controller.use_cfattn and (not crossattn):
            video_length = key.size()[0] // self.unet_chunk_size
            former_frame_index = [0] * video_length
            attn_mask = None
            if self.controller.attn_mask is not None:
                for m in self.controller.attn_mask:
                    if m.shape[1] == key.shape[1]:
                        attn_mask = m
            # BC * HW * 8D --> B * C * HW * 8D
            key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
            # B * C * HW * 8D --> B * C * HW * 8D
            if attn_mask is None:
                key = key[:, former_frame_index]
            else:
                key = repeat(key[:, attn_mask], "b d c -> b f d c", f=video_length)
            # B * C * HW * 8D --> BC * HW * 8D 
            key = rearrange(key, "b f d c -> (b f) d c").detach()
            value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
            if attn_mask is None:
                value = value[:, former_frame_index]
            else:
                value = repeat(value[:, attn_mask], "b d c -> b f d c", f=video_length)              
            value = rearrange(value, "b f d c -> (b f) d c").detach()
        
        # BC * HW * 8D --> BC * HW * 8 * D --> BC * 8 * HW * D
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # BC * 8 * HW2 * D
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # BC * 8 * HW2 * D2
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        '''for spatial-guided intra-frame attention'''
        if self.controller and self.controller.use_intraattn and (not crossattn): 
            ref_hidden_states = self.controller(None)
            assert ref_hidden_states.shape == encoder_hidden_states.shape
            query_ = attn.to_q(ref_hidden_states)
            key_ = attn.to_k(ref_hidden_states) 
            
            ''' 
            # for xformers implementation 
            if importlib.util.find_spec("xformers") is not None:
                # BC * HW * 8D --> BC * HW * 8 * D
                query_ = rearrange(query_, "b d (h c) -> b d h c", h=attn.heads)
                key_ = rearrange(key_, "b d (h c) -> b d h c", h=attn.heads)
                # BC * 8 * HW * D --> 8BC * HW * D
                query = rearrange(query, "b h d c -> b d h c")
                query = xformers.ops.memory_efficient_attention(
                    query_, key_ * self.sattn_scale_factor, query, 
                    attn_bias=torch.eye(query_.size(1), key_.size(1), 
                    dtype=query.dtype, device=query.device) * self.bias_weight, op=None
                )
                query = rearrange(query, "b d h c -> b h d c").detach()
            '''
            # BC * 8 * HW * D
            query_ = query_.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key_ = key_.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            query = F.scaled_dot_product_attention(
                query_, key_ * self.controller.intraattn_scale_factor, query, 
                attn_mask = torch.eye(query_.size(-2), key_.size(-2), 
                                      dtype=query.dtype, device=query.device) * self.controller.intraattn_bias,
            ).detach()
            #print('intra: ', GPU.getGPUs()[1].memoryUsed)
            del query_, key_
            torch.cuda.empty_cache()
        
        '''
        # for xformers implementation
        if importlib.util.find_spec("xformers") is not None:
            hidden_states = xformers.ops.memory_efficient_attention(
                    rearrange(query, "b h d c -> b d h c"), rearrange(key, "b h d c -> b d h c"), 
                    rearrange(value, "b h d c -> b d h c"), 
                    attn_bias=attention_mask, op=None
                )
            hidden_states = rearrange(hidden_states, "b d h c -> b h d c", h=attn.heads)
        '''
        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        # output: BC * 8 * HW * D2      
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        #print('cross: ', GPU.getGPUs()[1].memoryUsed)
        
        '''for temporal-guided inter-frame attention (FLATTEN)'''
        if self.controller and self.controller.use_interattn and (not crossattn):
            del query, key, value
            torch.cuda.empty_cache()
            bwd_mapping = None
            fwd_mapping = None
            flattn_mask = None
            for i, f in enumerate(self.controller.interattn_paras['fwd_mappings']):
                if f.shape[2] == hidden_states.shape[2]:
                    fwd_mapping = f
                    bwd_mapping = self.controller.interattn_paras['bwd_mappings'][i]
                    interattn_mask = self.controller.interattn_paras['interattn_masks'][i]
            video_length = key_raw.size()[0] // self.unet_chunk_size
            # BC * HW * 8D --> C * 8BD * HW
            key = rearrange(key_raw, "(b f) d c -> f (b c) d", f=video_length)
            query = rearrange(query_raw, "(b f) d c -> f (b c) d", f=video_length)
            # BC * 8 * HW * D --> C * 8BD * HW
            #key = rearrange(hidden_states, "(b f) h d c -> f (b h c) d", f=video_length) ########
            #query = rearrange(hidden_states, "(b f) h d c -> f (b h c) d", f=video_length) #######
            
            value = rearrange(hidden_states, "(b f) h d c -> f (b h c) d", f=video_length)
            key = torch.gather(key, 2, fwd_mapping.expand(-1,key.shape[1],-1))
            query = torch.gather(query, 2, fwd_mapping.expand(-1,query.shape[1],-1))
            value = torch.gather(value, 2, fwd_mapping.expand(-1,value.shape[1],-1))
            # C * 8BD * HW --> BHW, C, 8D
            key = rearrange(key, "f (b c) d -> (b d) f c", b=self.unet_chunk_size)
            query = rearrange(query, "f (b c) d -> (b d) f c", b=self.unet_chunk_size)
            value = rearrange(value, "f (b c) d -> (b d) f c", b=self.unet_chunk_size) 
            '''
            # for xformers implementation 
            if importlib.util.find_spec("xformers") is not None:
                # BHW * C * 8D --> BHW * C * 8 * D
                query = rearrange(query, "b d (h c) -> b d h c", h=attn.heads)
                key = rearrange(key, "b d (h c) -> b d h c", h=attn.heads)
                value = rearrange(value, "b d (h c) -> b d h c", h=attn.heads)
                B, D, C, _ = flattn_mask.shape
                C1 = int(np.ceil(C / 4) * 4)
                attn_bias = torch.zeros(B, D, C, C1, dtype=value.dtype, device=value.device) # HW * 1 * C * C
                attn_bias[:,:,:,:C].masked_fill_(interattn_mask.logical_not(), float("-inf")) # BHW * C * C
                hidden_states_ = xformers.ops.memory_efficient_attention(
                    query, key * self.controller.interattn_scale_factor, value, 
                    attn_bias=attn_bias.squeeze(1).repeat(self.unet_chunk_size*attn.heads,1,1)[:,:,:C], op=None
                )
                hidden_states_ = rearrange(hidden_states_, "b d h c -> b h d c", h=attn.heads).detach()
            '''
            # BHW * C * 8D --> BHW * C * 8 * D--> BHW * 8 * C * D
            query = query.view(-1, video_length, attn.heads, head_dim).transpose(1, 2).detach()
            key = key.view(-1, video_length, attn.heads, head_dim).transpose(1, 2).detach()
            value = value.view(-1, video_length, attn.heads, head_dim).transpose(1, 2).detach()
            hidden_states_ = F.scaled_dot_product_attention(
                query, key * self.controller.interattn_scale_factor, value, 
                attn_mask = (interattn_mask.repeat(self.unet_chunk_size,1,1,1))#.to(query.dtype)-1.0) * 1e6 -
                #torch.eye(interattn_mask.shape[2]).to(query.device).to(query.dtype) * 1e4,
            )
                
            # BHW * 8 * C * D --> C * 8BD * HW
            hidden_states_ = rearrange(hidden_states_, "(b d) h f c -> f (b h c) d", b=self.unet_chunk_size)
            hidden_states_ = torch.gather(hidden_states_, 2, bwd_mapping.expand(-1,hidden_states_.shape[1],-1)).detach()
            # C * 8BD * HW --> BC * 8 * HW * D
            hidden_states = rearrange(hidden_states_, "f (b h c) d -> (b f) h d c", b=self.unet_chunk_size, h=attn.heads)
            #print('inter: ', GPU.getGPUs()[1].memoryUsed)
            
        # BC * 8 * HW * D --> BC * HW * 8D
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def apply_FRESCO_attn(pipe):
    """
    Apply FRESCO-guided attention to a StableDiffusionPipeline
    """    
    frescoProc = FRESCOAttnProcessor2_0(2, AttentionControl())
    attnProc = AttnProcessor2_0()
    attn_processor_dict = {}
    for k in pipe.unet.attn_processors.keys():
        if k.startswith("up_blocks.2") or k.startswith("up_blocks.3"):
            attn_processor_dict[k] = frescoProc
        else:
            attn_processor_dict[k] = attnProc
    pipe.unet.set_attn_processor(attn_processor_dict)
    return frescoProc
    

"""
==========================================================================
PART II - FRESCO-based optimization
* optimize_feature(): function to optimze latent feature 
* my_forward(): hacked pipe.unet.forward(), adding feature optimization
* apply_FRESCO_opt(): function to apply FRESCO-based optimization to a StableDiffusionPipeline
* disable_FRESCO_opt(): function to disable the FRESCO-based optimization
==========================================================================
"""

def optimize_feature(sample, flows, occs, correlation_matrix=[], 
                     intra_weight = 1e2, iters=20, unet_chunk_size=2, optimize_temporal = True):
    """
    FRESO-guided latent feature optimization
    * optimize spatial correspondence (match correlation_matrix)
    * optimize temporal correspondence (match warped_image)
    """
    if (flows is None or occs is None or (not optimize_temporal)) and (intra_weight == 0 or len(correlation_matrix) == 0):
        return sample
    # flows=[fwd_flows, bwd_flows]: (N-1)*2*H1*W1
    # occs=[fwd_occs, bwd_occs]: (N-1)*H1*W1
    # sample: 2N*C*H*W
    torch.cuda.empty_cache()
    video_length = sample.shape[0] // unet_chunk_size
    latent = rearrange(sample.to(torch.float32), "(b f) c h w -> b f c h w", f=video_length)
    
    cs = torch.nn.Parameter((latent.detach().clone()))
    optimizer = torch.optim.Adam([cs], lr=0.2)

    # unify resolution
    if flows is not None and occs is not None:
        scale = sample.shape[2] * 1.0 / flows[0].shape[2]
        kernel = int(1 / scale)
        bwd_flow_ = F.interpolate(flows[1] * scale, scale_factor=scale, mode='bilinear').repeat(unet_chunk_size,1,1,1)
        bwd_occ_ = F.max_pool2d(occs[1].unsqueeze(1), kernel_size=kernel).repeat(unet_chunk_size,1,1,1) # 2(N-1)*1*H1*W1
        fwd_flow_ = F.interpolate(flows[0] * scale, scale_factor=scale, mode='bilinear').repeat(unet_chunk_size,1,1,1)
        fwd_occ_ = F.max_pool2d(occs[0].unsqueeze(1), kernel_size=kernel).repeat(unet_chunk_size,1,1,1) # 2(N-1)*1*H1*W1
        # match frame 0,1,2,3 and frame 1,2,3,0
        reshuffle_list = list(range(1,video_length))+[0]
        
    # attention_probs is the GRAM matrix of the normalized feature 
    attention_probs = None
    for tmp in correlation_matrix:
        if sample.shape[2] * sample.shape[3] == tmp.shape[1]:
            attention_probs = tmp # 2N*HW*HW
            break
     
    n_iter=[0]
    while n_iter[0] < iters:
        def closure():
            optimizer.zero_grad()
            
            loss = 0

            # temporal consistency loss 
            if optimize_temporal and flows is not None and occs is not None:
                c1 = rearrange(cs[:,:], "b f c h w -> (b f) c h w")
                c2 = rearrange(cs[:,reshuffle_list], "b f c h w -> (b f) c h w")
                warped_image1 = flow_warp(c1, bwd_flow_)
                warped_image2 = flow_warp(c2, fwd_flow_)
                loss = (abs((c2-warped_image1)*(1-bwd_occ_)) + abs((c1-warped_image2)*(1-fwd_occ_))).mean() * 2
                
            # spatial consistency loss
            if attention_probs is not None and intra_weight > 0:
                cs_vector = rearrange(cs, "b f c h w -> (b f) (h w) c")
                #attention_scores = torch.bmm(cs_vector, cs_vector.transpose(-1, -2))
                #cs_attention_probs = attention_scores.softmax(dim=-1)
                cs_vector = cs_vector / ((cs_vector ** 2).sum(dim=2, keepdims=True) ** 0.5)
                cs_attention_probs = torch.bmm(cs_vector, cs_vector.transpose(-1, -2))
                tmp = F.l1_loss(cs_attention_probs, attention_probs) * intra_weight
                loss = tmp + loss
                
            loss.backward()
            n_iter[0]+=1
            
            
            if False: # for debug
                print('Iteration: %d, loss: %f'%(n_iter[0]+1, loss.data.mean()))
            return loss
        optimizer.step(closure)

    torch.cuda.empty_cache()
    return adaptive_instance_normalization(rearrange(cs.data.to(sample.dtype), "b f c h w -> (b f) c h w"), sample)


def my_forward(self, steps = [], layers = [0,1,2,3], flows = None, occs = None, 
               correlation_matrix=[], intra_weight = 1e2, iters=20, optimize_temporal = True, saliency = None):
    """
    Hacked pipe.unet.forward()
    copied from https://github.com/huggingface/diffusers/blob/v0.19.3/src/diffusers/models/unet_2d_condition.py#L700
    if you are using a new version of diffusers, please copy the source code and modify it accordingly (find [HACK] in the code)
    * restore and return the decoder features
    * optimize the decoder features
    * perform background smoothing
    """    
    def forward(
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        The [`UNet2DConditionModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.FloatTensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            encoder_attention_mask (`torch.Tensor`):
                A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
                `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
                which adds large negative values to the attention scores corresponding to "discard" tokens.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttnProcessor`].
            added_cond_kwargs: (`dict`, *optional*):
                A kwargs dictionary containin additional embeddings that if specified are added to the embeddings that
                are passed along to the UNet blocks.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_2d_condition.UNet2DConditionOutput`] is returned, otherwise
                a `tuple` is returned where the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
        elif self.config.addition_embed_type == "text_image":
            # Kandinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

            image_embs = added_cond_kwargs.get("image_embeds")
            text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
            aug_emb = self.add_embedding(text_embs, image_embs)
        elif self.config.addition_embed_type == "text_time":
            # SDXL - style
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
        elif self.config.addition_embed_type == "image":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            aug_emb = self.add_embedding(image_embs)
        elif self.config.addition_embed_type == "image_hint":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            hint = added_cond_kwargs.get("hint")
            aug_emb, hint = self.add_embedding(image_embs, hint)
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
            # Kadinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )

            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(image_embeds)
        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        is_adapter = mid_block_additional_residual is None and down_block_additional_residuals is not None

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_block_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_block_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

                if is_adapter and len(down_block_additional_residuals) > 0:
                    sample += down_block_additional_residuals.pop(0)
            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )

        if is_controlnet:
            sample = sample + mid_block_additional_residual
        
        # 5. up
        '''
        [HACK] restore the decoder features in up_samples
        '''
        up_samples = ()
        #down_samples = ()
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            '''
            [HACK] restore the decoder features in up_samples
            [HACK] optimize the decoder features
            [HACK] perform background smoothing
            '''
            if i in layers:
                up_samples += (sample, )
            if timestep in steps and i in layers: 
                sample = optimize_feature(sample, flows, occs, correlation_matrix, 
                                          intra_weight, iters, optimize_temporal = optimize_temporal)
                if saliency is not None:
                    sample = warp_tensor(sample, flows, occs, saliency, 2)
            
            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        '''
        [HACK] return the output feature as well as the decoder features
        '''        
        if not return_dict:
            return (sample, ) + up_samples

        return UNet2DConditionOutput(sample=sample)
    
    return forward    


def apply_FRESCO_opt(pipe, steps = [], layers = [0,1,2,3], flows = None, occs = None, 
               correlation_matrix=[], intra_weight = 1e2, iters=20, optimize_temporal = True, saliency = None):
    """
    Apply FRESCO-based optimization to a StableDiffusionPipeline
    """  
    pipe.unet.forward = my_forward(pipe.unet, steps, layers, flows, occs, 
               correlation_matrix, intra_weight, iters, optimize_temporal, saliency)

def disable_FRESCO_opt(pipe):
    """
    Disable the FRESCO-based optimization
    """  
    apply_FRESCO_opt(pipe)


"""
=====================================================================================
PART III - Prepare parameters for FRESCO-guided attention/optimization 
* get_intraframe_paras(): get parameters for spatial-guided attention/optimization 
* get_flow_and_interframe_paras(): get parameters for temporal-guided attention/optimization 
=====================================================================================
"""

@torch.no_grad()
def get_intraframe_paras(pipe, imgs, frescoProc, 
                         prompt_embeds, do_classifier_free_guidance=True, seed=0):
    """
    Get parameters for spatial-guided attention and optimization
    * perform one step denoising 
    * collect attention feature, stored in frescoProc.controller.stored_attn['decoder_attn']
    * compute the gram matrix of the normalized feature for spatial consistency loss
    """

    noise_scheduler = pipe.scheduler 
    timestep = noise_scheduler.timesteps[-1]
    device = pipe._execution_device
    generator = torch.Generator(device=device).manual_seed(seed)
    B, C, H, W = imgs.shape
    
    frescoProc.controller.disable_controller()
    disable_FRESCO_opt(pipe)
    frescoProc.controller.clear_store()
    frescoProc.controller.enable_store()
    
    latents = pipe.prepare_latents(
        B,
        pipe.unet.config.in_channels,
        H,
        W,
        prompt_embeds.dtype,
        device,
        generator,
        latents = None,
    )

    latent_x0 = pipe.vae.config.scaling_factor * pipe.vae.encode(imgs.to(pipe.unet.dtype)).latent_dist.sample()
    latents = noise_scheduler.add_noise(latent_x0, latents, timestep).detach()

    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents 
    model_output = pipe.unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=prompt_embeds,
        cross_attention_kwargs=None,
        return_dict=False,
    )
    
    frescoProc.controller.disable_store()

    # gram matrix of the normalized feature for spatial consistency loss
    correlation_matrix = []
    for tmp in model_output[1:]:
        latent_vector = rearrange(tmp, "b c h w -> b (h w) c")
        latent_vector = latent_vector / ((latent_vector ** 2).sum(dim=2, keepdims=True) ** 0.5)
        attention_probs = torch.bmm(latent_vector, latent_vector.transpose(-1, -2))
        correlation_matrix += [attention_probs.detach().clone().to(torch.float32)]
        del attention_probs, latent_vector, tmp
    del model_output    
    
    gc.collect()
    torch.cuda.empty_cache()

    return correlation_matrix


@torch.no_grad()
def get_flow_and_interframe_paras(flow_model, imgs, visualize_pipeline=False):
    """
    Get parameters for temporal-guided attention and optimization
    * predict optical flow and occlusion mask
    * compute pixel index correspondence for FLATTEN
    """
    images = torch.stack([torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs], dim=0).cuda()
    imgs_torch = torch.cat([numpy2tensor(img) for img in imgs], dim=0)
    
    reshuffle_list = list(range(1,len(images)))+[0]
    
    results_dict = flow_model(images, images[reshuffle_list], attn_splits_list=[2], 
                              corr_radius_list=[-1], prop_radius_list=[-1], pred_bidir_flow=True)
    flow_pr = results_dict['flow_preds'][-1]  # [2*B, 2, H, W]
    fwd_flows, bwd_flows = flow_pr.chunk(2)   # [B, 2, H, W]
    fwd_occs, bwd_occs = forward_backward_consistency_check(fwd_flows, bwd_flows) # [B, H, W]
    
    warped_image1 = flow_warp(images, bwd_flows)
    bwd_occs = torch.clamp(bwd_occs + (abs(images[reshuffle_list]-warped_image1).mean(dim=1)>255*0.25).float(), 0 ,1)
    
    warped_image2 = flow_warp(images[reshuffle_list], fwd_flows)
    fwd_occs = torch.clamp(fwd_occs + (abs(images-warped_image2).mean(dim=1)>255*0.25).float(), 0 ,1)    
    
    if visualize_pipeline:
        print('visualized occlusion masks based on optical flows')
        viz = torchvision.utils.make_grid(imgs_torch * (1-fwd_occs.unsqueeze(1)), len(images), 1)
        visualize(viz.cpu(), 90)
        viz = torchvision.utils.make_grid(imgs_torch[reshuffle_list] * (1-bwd_occs.unsqueeze(1)), len(images), 1)
        visualize(viz.cpu(), 90) 
        
    attn_mask = []
    for scale in [8.0, 16.0, 32.0]:
        bwd_occs_ = F.interpolate(bwd_occs[:-1].unsqueeze(1), scale_factor=1./scale, mode='bilinear')
        attn_mask += [torch.cat((bwd_occs_[0:1].reshape(1,-1)>-1, bwd_occs_.reshape(bwd_occs_.shape[0],-1)>0.5), dim=0)]   
        
    fwd_mappings = []
    bwd_mappings = []
    interattn_masks = []
    for scale in [8.0, 16.0]:
        fwd_mapping, bwd_mapping, interattn_mask = get_mapping_ind(bwd_flows, bwd_occs, imgs_torch, scale=scale)
        fwd_mappings += [fwd_mapping]
        bwd_mappings += [bwd_mapping]
        interattn_masks += [interattn_mask]  
    
    interattn_paras = {}
    interattn_paras['fwd_mappings'] = fwd_mappings
    interattn_paras['bwd_mappings'] = bwd_mappings
    interattn_paras['interattn_masks'] = interattn_masks    

    gc.collect()
    torch.cuda.empty_cache()
    
    return [fwd_flows, bwd_flows], [fwd_occs, bwd_occs], attn_mask, interattn_paras
