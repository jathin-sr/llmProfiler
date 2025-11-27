def calculate_model_params(block_size, n_layer, n_embd):
    """Calculate approximate parameter count for GPT model"""
    vocab_size = 65
    
    # embedding parameters
    params = vocab_size * n_embd +block_size * n_embd
    
    # transformer blocks
    for _ in range(n_layer):
        # attention projections(Q, K, V, output)
        params += 4 * n_embd * n_embd
        # mlp layers
        params += n_embd * (4 * n_embd) + (4 * n_embd) * n_embd
        # layer norms(approx)
        params += 4 * n_embd
    
    # output projection
    params += n_embd * vocab_size
    
    return params