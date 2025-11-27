def build_training_command(config, model_cfg, batch_size, block_size, device):
    n_layer, n_head, n_embd = model_cfg
    
    base_cmd = [
        'python', 
        'train.py', 
        'config/train_shakespeare_char.py',
        f'--device={device}',
        f'--compile={config["training"]["compile"]}',
        f'--eval_iters={config["training"]["eval_iters"]}',
        f'--log_interval={config["training"]["log_interval"]}',
        f'--block_size={block_size}',
        f'--batch_size={batch_size}',
        f'--n_layer={n_layer}',
        f'--n_head={n_head}',
        f'--n_embd={n_embd}',
        f'--max_iters={config["training"]["max_iters"]}',
        f'--lr_decay_iters={config["training"]["lr_decay_iters"]}',
        f'--dropout={config["training"]["dropout"]}',
    ]
    
    return base_cmd