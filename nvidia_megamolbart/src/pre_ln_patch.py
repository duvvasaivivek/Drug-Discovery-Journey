import torch
import torch.nn as nn
from transformers.models.bart.modeling_bart import BartEncoderLayer, BartDecoderLayer

class PreLNEncoderLayer(BartEncoderLayer):
    def forward(self, hidden_states, attention_mask=None, layer_head_mask=None, output_attentions=False, **kwargs):
        if isinstance(hidden_states, tuple): hidden_states = hidden_states[0]
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = attn_outputs[0]
        attn_weights = attn_outputs[1] if output_attentions else None
        hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = torch.nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if output_attentions:
            return hidden_states, attn_weights
        return (hidden_states,)

class PreLNDecoderLayer(BartDecoderLayer):
    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, layer_head_mask=None, cross_attn_layer_head_mask=None, past_key_value=None, output_attentions=False, use_cache=True, **kwargs):
        if isinstance(hidden_states, tuple): hidden_states = hidden_states[0]
        if 'past_key_values' in kwargs: past_key_value = kwargs['past_key_values']
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        self_attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = self_attn_outputs[0]
        hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        cross_attn_outputs = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            cross_attn_outputs = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=None,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attn_outputs[0]
            hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = torch.nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if use_cache:
            outputs += (self_attn_outputs[1] if past_key_value is not None else None,)
        if output_attentions:
            outputs += (self_attn_outputs[2] if past_key_value is not None else None,)
            if encoder_hidden_states is not None:
                outputs += (cross_attn_outputs[2],)
        return outputs

def patch_bart_for_pre_ln(model):
    for layer in model.model.encoder.layers:
        layer.__class__ = PreLNEncoderLayer
    for layer in model.model.decoder.layers:
        layer.__class__ = PreLNDecoderLayer
        
    model.model.encoder.final_layernorm = nn.LayerNorm(model.config.d_model)
    model.model.decoder.final_layernorm = nn.LayerNorm(model.config.d_model)

    orig_enc_forward = model.model.encoder.forward
    def new_enc_forward(self, *args, **kwargs):
        res = orig_enc_forward(*args, **kwargs)
        if hasattr(res, 'last_hidden_state'):
            val = res.last_hidden_state
            if isinstance(val, tuple): val = val[0]
            res.last_hidden_state = self.final_layernorm(val)
            if 'last_hidden_state' in res: res['last_hidden_state'] = res.last_hidden_state
        elif isinstance(res, tuple):
            val = res[0]
            if isinstance(val, tuple): val = val[0]
            return (self.final_layernorm(val),) + res[1:]
        return res
    model.model.encoder.forward = new_enc_forward.__get__(model.model.encoder)
    
    orig_dec_forward = model.model.decoder.forward
    def new_dec_forward(self, *args, **kwargs):
        res = orig_dec_forward(*args, **kwargs)
        if hasattr(res, 'last_hidden_state'):
            val = res.last_hidden_state
            if isinstance(val, tuple): val = val[0]
            res.last_hidden_state = self.final_layernorm(val)
            if 'last_hidden_state' in res: res['last_hidden_state'] = res.last_hidden_state
        elif isinstance(res, tuple):
            val = res[0]
            if isinstance(val, tuple): val = val[0]
            return (self.final_layernorm(val),) + res[1:]
        return res
    model.model.decoder.forward = new_dec_forward.__get__(model.model.decoder)
