from utils.param_calcs import calculate_model_params
from utils.power_measurement import measure_training_power
from utils.parsing import parse_training_metrics
from utils.training_command import build_training_command

def run_profiling_experiment(config):
    components = [
        'forward_total', 'backward_total', 'optimizer_step', 'zero_grad', 'gradient_clipping',
        'embedding', 'transformer_blocks', 'final_layernorm', 'output_head', 'loss_calculation',
        'block_total', 'block_attention', 'block_mlp', 'block_layernorm1', 'block_layernorm2', 
        'block_residual1', 'block_residual2', 'attention_total', 'attention_qkv_proj', 
        'attention_reshape', 'attention_compute', 'attention_output', 'mlp_total', 
        'mlp_fc1', 'mlp_activation', 'mlp_fc2', 'mlp_dropout', 'io', 'first_data_load',
        'dynamic_learing_rate_loop', 'loss_evaluation', 'accumulation','data_loading',
        'profiling_n_printing', 'loss', 'mfu',
    ]
    
    results = []
    total_tests = (len(config['model_configs']) * len(config['batch_sizes']) * 
                   len(config['block_sizes']) * len(config['devices']))
    test_count = 0
    
    for model_cfg in config['model_configs']:
        for batch_size in config['batch_sizes']:
            for block_size in config['block_sizes']:
                for device in config['devices']:
                    test_count += 1
                    print(f"\n[{test_count}/{total_tests}] Testing: L{model_cfg[0]} H{model_cfg[1]} E{model_cfg[2]} Batch{batch_size} Block{block_size} {device.upper()}")
                    
                    training_cmd = build_training_command(config, model_cfg, batch_size, block_size, device)
                    param_count = calculate_model_params(block_size, model_cfg[0], model_cfg[2])
                    power_samples, training_metrics, training_time = measure_training_power(training_cmd)
                    result = process_experiment_results(model_cfg, batch_size, block_size, device, param_count, power_samples, training_metrics, training_time, components)
                    results.append(result)
    
    return results

def process_experiment_results(model_cfg, batch_size, block_size, device, param_count,
                             power_samples, training_metrics, training_time, components):
    n_layer, n_head, n_embd = model_cfg
    
    step_times, losses, profile_data = parse_training_metrics(training_metrics, components)
    
    avg_cpu = sum(s.get('cpu_power_mw', 0) for s in power_samples) / len(power_samples)
    avg_gpu = sum(s.get('gpu_power_mw', 0) for s in power_samples) / len(power_samples)
    total_power = avg_cpu + avg_gpu
    
    final_loss = losses[-1] if losses else float('inf')
    total_steps_time = sum(step_times) / 1000
    avg_step_time = total_steps_time / len(step_times) if step_times else float('inf')
    steps_per_second = len(step_times) / total_steps_time if total_steps_time > 0 else 0
    efficiency = steps_per_second / (total_power / 1000) if total_power > 0 else 0
    
    matmul_flops = 6 * param_count * block_size * batch_size
    attention_flops = 2 * batch_size * n_layer * (block_size * block_size * n_embd)
    total_flops_per_step = 2 * (matmul_flops + attention_flops)
    flops = total_flops_per_step * steps_per_second
    flops_per_watt = flops / total_power if total_power > 0 else 0
    
    result = {
        'device': device,
        'n_layer': n_layer,
        'n_head': n_head, 
        'n_embd': n_embd,
        'batch_size': batch_size,
        'block_size': block_size,
        'param_count_m': param_count / 1e6,
        'avg_cpu_power': avg_cpu,
        'avg_gpu_power': avg_gpu,
        'total_power': total_power,
        'final_loss': final_loss,
        'training_time': training_time,
        'avg_step_time': avg_step_time,
        'steps_per_second': steps_per_second,
        'steps_per_second_watt': efficiency,
        'flops_per_step': total_flops_per_step,
        'flops_per_second': flops,
        'flops_per_watt': flops_per_watt,
        'total_steps_time': total_steps_time,
    }
    
    for component in components:
        key = f"{component}_time_s"
        result[key] = sum(profile_data[component]) / len(profile_data[component]) if profile_data[component] else 0.0
    
    return result