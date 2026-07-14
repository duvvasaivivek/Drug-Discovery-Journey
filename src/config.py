import torch
from transformers import BartConfig

def get_megamolbart_config() -> BartConfig:
    """
    Returns the HuggingFace BartConfig matching the MegaMolBART model_config.yaml
    extracted from the .nemo file.
    """
    return BartConfig(
        vocab_size=512, # Will be overridden by actual tokenizer vocab size
        max_position_embeddings=512,
        encoder_layers=6,
        encoder_ffn_dim=2048,
        encoder_attention_heads=8,
        decoder_layers=6,
        decoder_ffn_dim=2048,
        decoder_attention_heads=8,
        d_model=512,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        activation_function="gelu",
        init_std=0.02,
        classifier_dropout=0.0,
        scale_embedding=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        is_encoder_decoder=True
    )

def map_nemo_to_huggingface(nemo_state_dict: dict, vocab_size: int) -> dict:
    """
    Translates the PyTorch state dict keys from NVIDIA NeMo format to 
    HuggingFace BartForConditionalGeneration format.
    """
    hf_state_dict = {}
    
    # 1. Embeddings
    hf_state_dict["model.shared.weight"] = nemo_state_dict["enc_dec_model.encoder_embedding.word_embeddings.weight"][:vocab_size]
    hf_state_dict["model.encoder.embed_tokens.weight"] = nemo_state_dict["enc_dec_model.encoder_embedding.word_embeddings.weight"][:vocab_size]
    hf_state_dict["model.decoder.embed_tokens.weight"] = nemo_state_dict["enc_dec_model.decoder_embedding.word_embeddings.weight"][:vocab_size]
    
    enc_pos = nemo_state_dict["enc_dec_model.encoder_embedding.position_embeddings.weight"]
    dec_pos = nemo_state_dict["enc_dec_model.decoder_embedding.position_embeddings.weight"]
    
    pad_tensor = torch.zeros(2, enc_pos.size(1), device=enc_pos.device, dtype=enc_pos.dtype)
    hf_state_dict["model.encoder.embed_positions.weight"] = torch.cat([pad_tensor, enc_pos], dim=0)
    hf_state_dict["model.decoder.embed_positions.weight"] = torch.cat([pad_tensor, dec_pos], dim=0)
    
    # 2. Encoder and Decoder layers
    for module in ["encoder", "decoder"]:
        for i in range(6): # 6 layers
            nemo_prefix = f"enc_dec_model.enc_dec_model.{module}.model.layers.{i}"
            hf_prefix = f"model.{module}.layers.{i}"
            
            # Layer Norms
            hf_state_dict[f"{hf_prefix}.self_attn_layer_norm.weight"] = nemo_state_dict[f"{nemo_prefix}.input_layernorm.weight"]
            hf_state_dict[f"{hf_prefix}.self_attn_layer_norm.bias"] = nemo_state_dict[f"{nemo_prefix}.input_layernorm.bias"]
            
            if module == "decoder":
                hf_state_dict[f"{hf_prefix}.encoder_attn_layer_norm.weight"] = nemo_state_dict[f"{nemo_prefix}.post_attention_layernorm.weight"]
                hf_state_dict[f"{hf_prefix}.encoder_attn_layer_norm.bias"] = nemo_state_dict[f"{nemo_prefix}.post_attention_layernorm.bias"]
                hf_state_dict[f"{hf_prefix}.final_layer_norm.weight"] = nemo_state_dict[f"{nemo_prefix}.post_inter_attention_layernorm.weight"]
                hf_state_dict[f"{hf_prefix}.final_layer_norm.bias"] = nemo_state_dict[f"{nemo_prefix}.post_inter_attention_layernorm.bias"]
            else:
                hf_state_dict[f"{hf_prefix}.final_layer_norm.weight"] = nemo_state_dict[f"{nemo_prefix}.post_attention_layernorm.weight"]
                hf_state_dict[f"{hf_prefix}.final_layer_norm.bias"] = nemo_state_dict[f"{nemo_prefix}.post_attention_layernorm.bias"]
            
            # MLP / FFN
            hf_state_dict[f"{hf_prefix}.fc1.weight"] = nemo_state_dict[f"{nemo_prefix}.mlp.dense_h_to_4h.weight"]
            hf_state_dict[f"{hf_prefix}.fc1.bias"] = nemo_state_dict[f"{nemo_prefix}.mlp.dense_h_to_4h.bias"]
            hf_state_dict[f"{hf_prefix}.fc2.weight"] = nemo_state_dict[f"{nemo_prefix}.mlp.dense_4h_to_h.weight"]
            hf_state_dict[f"{hf_prefix}.fc2.bias"] = nemo_state_dict[f"{nemo_prefix}.mlp.dense_4h_to_h.bias"]
            
            # Self Attention
            qkv_weight = nemo_state_dict[f"{nemo_prefix}.self_attention.query_key_value.weight"]
            qkv_bias = nemo_state_dict[f"{nemo_prefix}.self_attention.query_key_value.bias"]
            
            q_w, k_w, v_w = torch.chunk(qkv_weight, 3, dim=0)
            q_b, k_b, v_b = torch.chunk(qkv_bias, 3, dim=0)
            
            hf_state_dict[f"{hf_prefix}.self_attn.q_proj.weight"] = q_w
            hf_state_dict[f"{hf_prefix}.self_attn.q_proj.bias"] = q_b
            hf_state_dict[f"{hf_prefix}.self_attn.k_proj.weight"] = k_w
            hf_state_dict[f"{hf_prefix}.self_attn.k_proj.bias"] = k_b
            hf_state_dict[f"{hf_prefix}.self_attn.v_proj.weight"] = v_w
            hf_state_dict[f"{hf_prefix}.self_attn.v_proj.bias"] = v_b
            
            hf_state_dict[f"{hf_prefix}.self_attn.out_proj.weight"] = nemo_state_dict[f"{nemo_prefix}.self_attention.dense.weight"]
            hf_state_dict[f"{hf_prefix}.self_attn.out_proj.bias"] = nemo_state_dict[f"{nemo_prefix}.self_attention.dense.bias"]
            
            # Cross Attention (Decoder only)
            if module == "decoder":
                cq_w = nemo_state_dict[f"{nemo_prefix}.inter_attention.query.weight"]
                cq_b = nemo_state_dict[f"{nemo_prefix}.inter_attention.query.bias"]
                
                ckv_weight = nemo_state_dict[f"{nemo_prefix}.inter_attention.key_value.weight"]
                ckv_bias = nemo_state_dict[f"{nemo_prefix}.inter_attention.key_value.bias"]
                
                ck_w, cv_w = torch.chunk(ckv_weight, 2, dim=0)
                ck_b, cv_b = torch.chunk(ckv_bias, 2, dim=0)
                
                hf_state_dict[f"{hf_prefix}.encoder_attn.q_proj.weight"] = cq_w
                hf_state_dict[f"{hf_prefix}.encoder_attn.q_proj.bias"] = cq_b
                hf_state_dict[f"{hf_prefix}.encoder_attn.k_proj.weight"] = ck_w
                hf_state_dict[f"{hf_prefix}.encoder_attn.k_proj.bias"] = ck_b
                hf_state_dict[f"{hf_prefix}.encoder_attn.v_proj.weight"] = cv_w
                hf_state_dict[f"{hf_prefix}.encoder_attn.v_proj.bias"] = cv_b
                
                hf_state_dict[f"{hf_prefix}.encoder_attn.out_proj.weight"] = nemo_state_dict[f"{nemo_prefix}.inter_attention.dense.weight"]
                hf_state_dict[f"{hf_prefix}.encoder_attn.out_proj.bias"] = nemo_state_dict[f"{nemo_prefix}.inter_attention.dense.bias"]

    # 3. LM Head
    hf_state_dict["lm_head.weight"] = nemo_state_dict["enc_dec_model.decoder_embedding.word_embeddings.weight"][:vocab_size]
    
    # 4. Final Layer Norms (Encoder/Decoder)
    if "enc_dec_model.enc_dec_model.encoder.final_layernorm.weight" in nemo_state_dict:
        hf_state_dict["model.encoder.layer_norm.weight"] = nemo_state_dict["enc_dec_model.enc_dec_model.encoder.final_layernorm.weight"]
        hf_state_dict["model.encoder.layer_norm.bias"] = nemo_state_dict["enc_dec_model.enc_dec_model.encoder.final_layernorm.bias"]
    if "enc_dec_model.enc_dec_model.decoder.final_layernorm.weight" in nemo_state_dict:
        hf_state_dict["model.decoder.layer_norm.weight"] = nemo_state_dict["enc_dec_model.enc_dec_model.decoder.final_layernorm.weight"]
        hf_state_dict["model.decoder.layer_norm.bias"] = nemo_state_dict["enc_dec_model.enc_dec_model.decoder.final_layernorm.bias"]
        
    return hf_state_dict
