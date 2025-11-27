import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, root_dir)
import utils.plot_helper as plot_helper
load_latest_results = plot_helper.load_latest_results
df, latest_file = load_latest_results()

def plot_flops_per_watt_vs_batch_size(df, save_path=None):
    """
    Plot 4A: FLOPs/watt vs Batch Size as double bar chart
    """
    # Focus on one representative model configuration for clarity
    plot_data = df[
        (df['n_layer'] == 4) & 
        (df['n_head'] == 4) & 
        (df['n_embd'] == 256) &
        (df['block_size'] == 64)
    ].copy()
    
    if 'flops_per_watt' not in df.columns:
        plot_data['total_power'] = plot_data['avg_cpu_power'] + plot_data['avg_gpu_power']
        # Using flops_per_second as proxy for computational throughput
        plot_data['flops_per_watt'] = plot_data['flops_per_second'] / plot_data['total_power'] /10**9
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Get batch sizes and devices
    batch_sizes = sorted(plot_data['batch_size'].unique())
    devices = ['cpu', 'mps']
    
    # Colors
    cpu_color = '#1f77b4'  # Blue
    mps_color = '#ff7f0e'  # Orange
    
    # Bar settings
    x = np.arange(len(batch_sizes))
    width = 0.35
    
    # Calculate y_max for proper scaling
    y_max = plot_data['flops_per_watt'].max() * 1.2
    
    for i, device in enumerate(devices):
        device_data = plot_data[plot_data['device'] == device].sort_values('batch_size')
        
        if not device_data.empty:
            flops_per_watt = device_data['flops_per_watt'].values
            x_pos = x + i * width
            
            # Double bars
            bars = ax.bar(x_pos, flops_per_watt, width, 
                         color=cpu_color if device == 'cpu' else mps_color, 
                         alpha=0.8, label=f"device='{device}'")
            
            # Add value labels on the bars
            for j, value in enumerate(flops_per_watt):
                ax.text(x_pos[j], value + y_max*0.01, f'{value/10**9:.2f}', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Create grouped x-axis labels
    x_labels = [f"{bs}" for bs in batch_sizes]
    
    # Set main x-axis labels
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(x_labels, fontsize=11, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('GFLOPs per Watt', fontsize=12, fontweight='bold')
    ax.set_title('Processor Efficiency vs Batch Size\n(Model: 4 layers, 4 heads, 256 embd, 64 block)', 
                 fontsize=14, fontweight='bold')
    
    # Add legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Set the y-limits
    ax.set_ylim(0, y_max)
    ax.set_xlim(-0.5, len(batch_sizes) - 0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    return fig

def plot_flops_per_watt_vs_block_size(df, save_path=None):
    """
    Plot 4B: FLOPs per Watt vs Block Size
    Double bar chart comparing computational efficiency between CPU and MPS devices
    """
    # Focus on one representative model size
    plot_data = df[
        (df['n_layer'] == 4) & 
        (df['n_head'] == 4) & 
        (df['n_embd'] == 256) &
        (df['batch_size'] == 32)
    ].copy()
    
    # Calculate FLOPs per watt
    plot_data['total_power'] = plot_data['avg_cpu_power'] + plot_data['avg_gpu_power']
    plot_data['flops_per_watt'] = plot_data['flops_per_second'] / plot_data['total_power'] /10**9
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Get block sizes and devices
    block_sizes = sorted(plot_data['block_size'].unique())
    devices = ['cpu', 'mps']
    
    # Colors
    cpu_color = '#1f77b4'  # Blue
    mps_color = '#ff7f0e'  # Orange
    
    # Bar settings
    x = np.arange(len(block_sizes))
    width = 0.35
    
    # Calculate y_max for proper scaling
    y_max = plot_data['flops_per_watt'].max() * 1.2
    
    for i, device in enumerate(devices):
        device_data = plot_data[plot_data['device'] == device].sort_values('block_size')
        
        if not device_data.empty:
            flops_per_watt = device_data['flops_per_watt'].values
            x_pos = x + i * width
            
            # Double bars
            bars = ax.bar(x_pos, flops_per_watt, width, 
                         color=cpu_color if device == 'cpu' else mps_color, 
                         alpha=0.8, label=f"device='{device}'")
            
            # Add value labels on the bars
            for j, value in enumerate(flops_per_watt):
                ax.text(x_pos[j], value + y_max*0.01, f'{value:.2f}', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Create grouped x-axis labels
    x_labels = [f"{bs}" for bs in block_sizes]
    
    # Set main x-axis labels
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(x_labels, fontsize=11, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Block Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('GFLOPs per Watt', fontsize=12, fontweight='bold')
    ax.set_title('Processor Efficiency vs Block Size\n (Model: 4 layers, 4 heads, 256 embd, 32 batch)', 
                 fontsize=14, fontweight='bold')
    
    # Add legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Set the y-limits
    ax.set_ylim(0, y_max)
    ax.set_xlim(-0.5, len(block_sizes) - 0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    return fig

def plot_flops_per_watt_vs_model_arch(df, save_path=None):
    """
    Plot 4C: FLOPs per Watt vs Model Architecture
    Double bar chart comparing computational efficiency between CPU and MPS devices
    """
    # Focus on specific model architectures with same batch_size and block_size
    target_archs = [
        (2, 2, 128),   # L2H2E128
        (4, 4, 256),   # L4H4E256  
        (6, 6, 384)    # L6H6E384
    ]
    
    plot_data = df[
        (df['batch_size'] == 2) & 
        (df['block_size'] == 128)
    ].copy()
    
    # Filter for target architectures
    arch_filter = False
    for n_layer, n_head, n_embd in target_archs:
        arch_filter |= (
            (plot_data['n_layer'] == n_layer) & 
            (plot_data['n_head'] == n_head) & 
            (plot_data['n_embd'] == n_embd)
        )
    
    plot_data = plot_data[arch_filter].copy()
    
    # Calculate FLOPs per watt
    plot_data['total_power'] = plot_data['avg_cpu_power'] + plot_data['avg_gpu_power']
    plot_data['flops_per_watt'] = plot_data['flops_per_second'] / plot_data['total_power'] /10**9
    
    # Create model architecture labels
    def get_model_label(row):
        params_m = row['param_count_m']
        return f"L{row['n_layer']}H{row['n_head']}E{row['n_embd']}\n(~{params_m:.1f}M)"
    
    plot_data['model_label'] = plot_data.apply(get_model_label, axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get model architectures in specified order
    model_archs = []
    for n_layer, n_head, n_embd in target_archs:
        label = f"L{n_layer}H{n_head}E{n_embd}"
        # Find the corresponding label with parameter count
        matching_data = plot_data[
            (plot_data['n_layer'] == n_layer) & 
            (plot_data['n_head'] == n_head) & 
            (plot_data['n_embd'] == n_embd)
        ]
        if not matching_data.empty:
            model_archs.append(matching_data['model_label'].iloc[0])
    
    devices = ['cpu', 'mps']
    
    # Colors
    cpu_color = '#1f77b4'  # Blue
    mps_color = '#ff7f0e'  # Orange
    
    # Bar settings
    x = np.arange(len(model_archs))
    width = 0.35
    
    # Calculate y_max for proper scaling
    y_max = plot_data['flops_per_watt'].max() * 1.2
    
    for i, device in enumerate(devices):
        device_data = plot_data[plot_data['device'] == device]
        
        # Ensure correct order
        flops_per_watt = []
        for arch in model_archs:
            arch_data = device_data[device_data['model_label'] == arch]
            if not arch_data.empty:
                flops_per_watt.append(arch_data['flops_per_watt'].iloc[0])
            else:
                flops_per_watt.append(0)
        
        x_pos = x + i * width
        
        # Double bars
        bars = ax.bar(x_pos, flops_per_watt, width, 
                     color=cpu_color if device == 'cpu' else mps_color, 
                     alpha=0.8, label=f"device='{device}'")
        
        # Add value labels on the bars
        for j, value in enumerate(flops_per_watt):
            if value > 0:
                ax.text(x_pos[j], value + y_max*0.01, f'{value:.2f}', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Set x-axis labels
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(model_archs, fontsize=10, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
    ax.set_ylabel('GFLOPs per Watt', fontsize=12, fontweight='bold')
    ax.set_title('Processor Efficiency vs Model Architecture\n(Batch size: 32, Block size: 64)', 
                 fontsize=14, fontweight='bold')
    
    # Add legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Set the y-limits
    ax.set_ylim(0, y_max)
    ax.set_xlim(-0.5, len(model_archs) - 0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    return fig

fig_4a = plot_flops_per_watt_vs_batch_size(df, save_path='results/plot_4a_flops_per_watt_vs_batch_size.png')
fig_4b = plot_flops_per_watt_vs_block_size(df, save_path='results/plot_4b_flops_per_watt_vs_block_size.png')
fig_4c = plot_flops_per_watt_vs_model_arch(df, save_path='results/plot_4c_flops_per_watt_vs_model_arch.png')
