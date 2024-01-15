from importlib import import_module
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import einsum, nn
from einops import rearrange   

from diffusers.utils import USE_PEFT_BACKEND
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import Attention


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

class SelfSubjectAttnProcessor:
    r"""
    processor for performing self subject attention-related computations.
    """
    
    def __init__(self, omega_ref = 3.0):
        self.omega_ref = omega_ref
        self.cached_j = {}
        
    def get_self_subject_attention_scores(
        self, query: torch.Tensor, key: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype
        
        alpha =  query.shape[2] ** (-0.5)

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            # self subject attention mask
            baddbmm_input = torch.log(attention_mask)
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=alpha,
        )
        del baddbmm_input

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        reference_hidden_states = None
        if isinstance(hidden_states, tuple):
            hidden_states, reference_hidden_states = hidden_states
            attention_mask, self_subject_attention_mask = attention_mask
                
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)
            
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
            if reference_hidden_states is not None:
                reference_hidden_states = attn.spatial_norm(reference_hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            if reference_hidden_states is not None:
                reference_hidden_states = reference_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            if reference_hidden_states is not None:
                reference_hidden_states = attn.group_norm(reference_hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
                
        if reference_hidden_states is not None:
            reference_key = attn.to_k(reference_hidden_states, *args)
            reference_value = attn.to_v(reference_hidden_states, *args)

            reference_key = attn.head_to_batch_dim(reference_key)
            reference_value = attn.head_to_batch_dim(reference_value)
            
            key = torch.cat([key, reference_key], dim=1)
            value = torch.cat([value, reference_value], dim=1)
            
            # quety (batch, seq_len, dim)
            # key (batch, seq_len x 2, dim)
            # value (batch, seq_len x 2, dim)
            
            # q * k^T (batch, seq_len, seq_len x 2)
            # mask (batch, seq_len, seq_len x 2)
            
            cache_key = f"{query.shape[0]}_{query.shape[1]}_{query.shape[1]}"
            head_dim = query.shape[0] // batch_size
            
            if cache_key in self.cached_j:
                j = self.cached_j[cache_key]
            else:            
                j = torch.ones((query.shape[0], query.shape[1], query.shape[1]), device=query.device, dtype=query.dtype)
                self.cached_j[cache_key] = j
                
            mask_size = int(sequence_length ** 0.5)
            ss_mask = F.interpolate(self_subject_attention_mask, size=(mask_size, mask_size), mode='bilinear', align_corners=False)
            ss_mask = ss_mask.view(-1, sequence_length)
            # (batch, 1, sequence_length, sequence_length)
            ss_mask = ss_mask.unsqueeze(1).repeat(1, 1, sequence_length, 1)
            ss_mask = ss_mask.repeat(1, head_dim, 1, 1)
            ss_mask = rearrange(ss_mask, 'b h s d -> (b h) s d')
            
            mask_ref = self.omega_ref * ss_mask
            
            # attention mask (batch * heads, seq, dim * 2 )
            attention_mask = torch.cat([j, mask_ref], dim=2)
                        
            # # TODO: support reference mask
            # if attention_mask is not None:
            #     attention_mask = torch.cat([attention_mask, attention_mask], dim=1)          
                
        if reference_hidden_states is not None:
            attention_probs = self.get_self_subject_attention_scores(query, key, attention_mask)
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states