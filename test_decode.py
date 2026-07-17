import torch
import torch.nn as nn
import types
from nvidia_megamolbart.src.loader import load_model
from nvidia_megamolbart.src.generation import generate_molecules

# Load model
model, tok = load_model('nvidia_megamolbart/models/megamolbart')

# Patch for Pre-LN
def pre_ln_enc_forward(self, hidden_states, attention_mask, **kwargs):
    if isinstance(hidden_states, tuple): hidden_states = hidden_states[0]
    residual = hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)
    
    # Check if there are other kwargs like layer_head_mask
    if 'layer_head_mask' in kwargs:
        hidden_states, _ = self.self_attn(hidden_states, attention_mask=attention_mask, layer_head_mask=kwargs.get('layer_head_mask'))
    else:
        hidden_states, _ = self.self_attn(hidden_states, attention_mask=attention_mask)
        
    hidden_states = residual + hidden_states
    
    residual = hidden_states
    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.activation_fn(self.fc1(hidden_states))
    hidden_states = self.fc2(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states

def pre_ln_dec_forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, layer_head_mask=None, cross_attn_layer_head_mask=None, past_key_value=None, output_attentions=False, use_cache=True, **kwargs):
    if 'past_key_values' in kwargs:
        past_key_value = kwargs['past_key_values']
    if isinstance(hidden_states, tuple): hidden_states = hidden_states[0]
    residual = hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)
    
    # Self Attention
    self_attn_outputs = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    hidden_states = self_attn_outputs[0]
    hidden_states = residual + hidden_states
    
    # Cross Attention
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
        hidden_states = residual + hidden_states

    # MLP
    residual = hidden_states
    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.activation_fn(self.fc1(hidden_states))
    hidden_states = self.fc2(hidden_states)
    hidden_states = residual + hidden_states
    
    outputs = (hidden_states,)
    if use_cache:
        outputs += (self_attn_outputs[1] if past_key_value is not None else None,)
    if output_attentions:
        outputs += (self_attn_outputs[2] if past_key_value is not None else None,)
        if encoder_hidden_states is not None:
            outputs += (cross_attn_outputs[2],)
    return outputs

for layer in model.model.encoder.layers:
    layer.forward = types.MethodType(pre_ln_enc_forward, layer)
for layer in model.model.decoder.layers:
    layer.forward = types.MethodType(pre_ln_dec_forward, layer)

# Also apply the missing final layer norms for encoder and decoder outputs!
enc_final_ln = nn.LayerNorm(512)
dec_final_ln = nn.LayerNorm(512)

# Load their weights from the NeMo state dict!
raw_ckpt = torch.load('nvidia_megamolbart/models/megamolbart/model_weights.ckpt', map_location='cpu')
sd = raw_ckpt.get('state_dict', raw_ckpt)
enc_final_ln.weight.data = sd['enc_dec_model.enc_dec_model.encoder.model.final_layernorm.weight']
enc_final_ln.bias.data = sd['enc_dec_model.enc_dec_model.encoder.model.final_layernorm.bias']
dec_final_ln.weight.data = sd['enc_dec_model.enc_dec_model.decoder.model.final_layernorm.weight']
dec_final_ln.bias.data = sd['enc_dec_model.enc_dec_model.decoder.model.final_layernorm.bias']

def safe_ln(x, ln):
    if isinstance(x, tuple): return (ln(x[0]),) + x[1:]
    return ln(x)

class PreLNHead(nn.Module):
    def __init__(self, ln, head):
        super().__init__()
        self.ln = ln
        self.head = head
    def forward(self, x):
        return self.head(self.ln(x))

model.lm_head = PreLNHead(dec_final_ln, model.lm_head)

orig_enc_forward = model.model.encoder.forward
def new_enc_forward(self, *args, **kwargs):
    res = orig_enc_forward(*args, **kwargs)
    res['last_hidden_state'] = enc_final_ln(res['last_hidden_state'])
    return res
model.model.encoder.forward = types.MethodType(new_enc_forward, model.model.encoder)

smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
input_ids = torch.tensor([tok.encode(smiles)], dtype=torch.long, device=model.device)
mask = torch.ones_like(input_ids)
enc_out = model.get_encoder()(input_ids=input_ids, attention_mask=mask, return_dict=True)
print('DECODING BASE WITH PRE-LN PATCH...')
print(generate_molecules(enc_out.last_hidden_state, mask, model, tok, max_length=50))
