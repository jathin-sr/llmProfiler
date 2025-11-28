import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, root_dir)
import utils.plot_helper as plot_helper
load_latest_results = plot_helper.load_latest_results
df, latest_file = load_latest_results()

def plot_energy_vs_batch_size(df, save_path=None):
    """
    Plot 2A: Energy consumption across batch sizes with double bars
    """
    # Focus on one representative model size
    plot_data = df[
        (df['n_layer'] == 4) & 
        (df['n_head'] == 4) & 
        (df['n_embd'] == 256) &
        (df['block_size'] == 64)
    ].copy()
    
    # Calculate energy consumption (Joules)
    plot_data['total_energy'] = (plot_data['avg_cpu_power'] + plot_data['avg_gpu_power']) * plot_data['training_time']
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Get batch sizes and devices
    batch_sizes = sorted(plot_data['batch_size'].unique())
    devices = ['cpu', 'mps']
    
    # Colors - Blue for CPU, Orange for GPU
    cpu_color = '#1f77b4'  # Blue
    mps_color = '#ff7f0e'  # Orange
    
    # Bar settings
    x = np.arange(len(batch_sizes))
    width = 0.35
    
    # Calculate y_max for proper scaling
    y_max = plot_data['total_energy'].max() * 1.2
    
    for i, device in enumerate(devices):
        device_data = plot_data[plot_data['device'] == device].sort_values('batch_size')
        
        if not device_data.empty:
            energy = device_data['total_energy'].values
            x_pos = x + i * width
            
            # Single bars (no stacking)
            bars = ax.bar(x_pos, energy, width, 
                         color=cpu_color if device == 'cpu' else mps_color, 
                         alpha=0.8, label=f"device='{device}'")
            
            # Add energy value labels on the bars
            for j, energy_val in enumerate(energy):
                ax.text(x_pos[j], energy_val + y_max*0.01, f'{energy_val:.0f}J', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Create grouped x-axis labels
    x_labels = []
    for i, batch_size in enumerate(batch_sizes):
        label = f"{batch_size}"
        x_labels.append(label)
    
    # Set main x-axis labels (batch sizes centered under each group)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(x_labels, fontsize=11, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Energy Consumption (Joules)', fontsize=12, fontweight='bold')
    ax.set_title('Energy Consumption vs Batch Size\n(Model: 4 layers, 4 heads, 256 embd, 64 block)', 
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

def plot_energy_vs_block_size(df, save_path=None):
    """
    Plot 2B: Energy consumption across block sizes with double bars
    """
    # Focus on one representative model size
    plot_data = df[
        (df['n_layer'] == 4) & 
        (df['n_head'] == 4) & 
        (df['n_embd'] == 256) &
        (df['batch_size'] == 32)
    ].copy()
    
    # Calculate energy consumption (Joules)
    plot_data['total_energy'] = (plot_data['avg_cpu_power'] + plot_data['avg_gpu_power']) * plot_data['training_time']
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Get block sizes and devices
    block_sizes = sorted(plot_data['block_size'].unique())
    devices = ['cpu', 'mps']
    
    # Colors - Blue for CPU, Orange for GPU
    cpu_color = '#1f77b4'  # Blue
    mps_color = '#ff7f0e'  # Orange
    
    # Bar settings
    x = np.arange(len(block_sizes))
    width = 0.35
    
    # Calculate y_max for proper scaling
    y_max = plot_data['total_energy'].max() * 1.2
    
    for i, device in enumerate(devices):
        device_data = plot_data[plot_data['device'] == device].sort_values('block_size')
        
        if not device_data.empty:
            energy = device_data['total_energy'].values
            x_pos = x + i * width
            
            # Single bars (no stacking)
            bars = ax.bar(x_pos, energy, width, 
                         color=cpu_color if device == 'cpu' else mps_color, 
                         alpha=0.8, label=f"device='{device}'")
            
            # Add energy value labels on the bars
            for j, energy_val in enumerate(energy):
                ax.text(x_pos[j], energy_val + y_max*0.01, f'{energy_val:.0f}J', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Create grouped x-axis labels
    x_labels = []
    for i, block_size in enumerate(block_sizes):
        label = f"{block_size}"
        x_labels.append(label)
    
    # Set main x-axis labels (block sizes centered under each group)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(x_labels, fontsize=11, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Block Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Energy Consumption (Joules)', fontsize=12, fontweight='bold')
    ax.set_title('Energy Consumption vs Block Size\n(Model: 4 layers, 4 heads, 256 embd, 32 batch)', 
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

def plot_energy_vs_model_architecture(df, save_path=None):
    """
    Plot 2C: Energy consumption across model architectures with double bars
    """
    # Focus on one representative batch size
    plot_data = df[
        (df['batch_size'] == 32) & 
        (df['block_size'] == 64)
    ].copy()
    
    # Calculate energy consumption (Joules)
    plot_data['total_energy'] = (plot_data['avg_cpu_power'] + plot_data['avg_gpu_power']) * plot_data['training_time']
    
    # Create model architecture labels
    def get_model_label(row):
        params_m = row['param_count_m']
        return f"L{row['n_layer']}H{row['n_head']}E{row['n_embd']}"
    
    plot_data['model_label'] = plot_data.apply(get_model_label, axis=1)
    
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Get unique model architectures sorted by parameter count
    model_archs = plot_data['model_label'].unique()
    
    devices = ['cpu', 'mps']
    
    # Colors - Blue for CPU, Orange for GPU
    cpu_color = '#1f77b4'  # Blue
    mps_color = '#ff7f0e'  # Orange
    
    # Bar settings
    x = np.arange(len(model_archs))
    width = 0.35
    
    # Calculate y_max for proper scaling
    y_max = plot_data['total_energy'].max() * 1.2
    
    for i, device in enumerate(devices):
        device_data = plot_data[plot_data['device'] == device]
        
        # Ensure correct order
        device_data = device_data.set_index('model_label').reindex(model_archs).reset_index()
        
        energy = device_data['total_energy'].fillna(0).values
        x_pos = x + i * width
        
        # Single bars (no stacking)
        bars = ax.bar(x_pos, energy, width, 
                     color=cpu_color if device == 'cpu' else mps_color, 
                     alpha=0.8, label=f"device='{device}'")
        
        # Add energy value labels on the bars
        for j, energy_val in enumerate(energy):
            if energy_val > 0:  # Only label if there's data
                ax.text(x_pos[j], energy_val + y_max*0.01, f'{energy_val:.0f}J', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Set x-axis labels
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(model_archs, fontsize=10, ha='right')
    
    # Customize plot
    ax.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
    ax.set_ylabel('Energy Consumption (Joules)', fontsize=12, fontweight='bold')
    ax.set_title('Energy Consumption vs Model Architecture\n(Batch Size: 32, Block Size: 64)', 
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

fig_2a = plot_energy_vs_batch_size(df, save_path='results/plot_2a_energy_vs_batch_size.png')
fig_2b = plot_energy_vs_block_size(df, save_path='results/plot_2b_energy_vs_block_size.png')
fig_2c = plot_energy_vs_model_architecture(df, save_path='results/plot_2c_energy_vs_model_arch.png')