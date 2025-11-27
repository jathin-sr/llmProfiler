import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, root_dir)
import utils.plot_helper as plot_helper
load_latest_results = plot_helper.load_latest_results
df, latest_file = load_latest_results()

def plot_power_vs_batch_size_enhanced(df, save_path=None):
    """
    Enhanced Plot 1A: Power composition across batch sizes with proper grouped labeling
    """
    # Focus on one representative model size
    plot_data = df[
        (df['n_layer'] == 4) & 
        (df['n_head'] == 4) & 
        (df['n_embd'] == 256) &
        (df['block_size'] == 64)
    ].copy()
    
    fig, ax = plt.subplots(figsize=(26, 9))
    
    # Get batch sizes and devices
    batch_sizes = sorted(plot_data['batch_size'].unique())
    devices = ['cpu', 'mps']
    
    # Colors - Red for CPU, Green for GPU
    cpu_color = '#d62728'  # Red
    gpu_color = '#2ca02c'  # Green
    
    # Bar settings
    x = np.arange(len(batch_sizes))
    width = 0.35
    
    # Arrow configuration
    arrow_offset = 0.25
    arrow_props = dict(arrowstyle='->', color='gray', lw=1.5, alpha=0.7)
    
    # Calculate y_max BEFORE using it
    y_max = max(plot_data['avg_cpu_power'] + plot_data['avg_gpu_power']) * 1.3
    
    for i, device in enumerate(devices):
        device_data = plot_data[plot_data['device'] == device].sort_values('batch_size')
        
        if not device_data.empty:
            cpu_power = device_data['avg_cpu_power'].values
            gpu_power = device_data['avg_gpu_power'].values
            
            x_pos = x + i * width
            
            # Stacked bars
            bars_cpu = ax.bar(x_pos, cpu_power, width, 
                             color=cpu_color, alpha=0.8)
            bars_gpu = ax.bar(x_pos, gpu_power, width, bottom=cpu_power,
                             color=gpu_color, alpha=0.8)
            
            # Add power value labels on the bars
            for j, (cpu, gpu) in enumerate(zip(cpu_power, gpu_power)):
                # CPU power label (centered in CPU section)
                if cpu > 10:
                    ax.text(x_pos[j], cpu/2, f'CPU: {cpu:.0f}W', 
                           ha='center', va='center', fontsize=8, fontweight='bold',
                           color='white')
                elif cpu > 0:
                    ax.annotate(f'CPU: {cpu:.0f}W', 
                               xy=(x_pos[j], cpu/2),
                               xytext=(x_pos[j] - arrow_offset, cpu/2 + 50),
                               arrowprops=arrow_props,
                               fontsize=7, fontweight='bold', ha='right')
                
                # GPU power label (centered in GPU section)
                gpu_center = cpu + gpu/2
                if gpu > 100:
                    ax.text(x_pos[j], gpu_center, f'GPU: {gpu:.0f}W', 
                           ha='center', va='center', fontsize=8, fontweight='bold',
                           color='white')
                elif gpu > 0:
                    ax.annotate(f'GPU: {gpu:.2f}W', 
                               xy=(x_pos[j], gpu_center),
                               xytext=(x_pos[j] - arrow_offset, gpu_center + 50),
                               arrowprops=arrow_props,
                               fontsize=7, fontweight='bold', ha='right')
                
                # Total power label at top
                total_power = cpu + gpu
                ax.text(x_pos[j], total_power + 15, f'Total: {total_power:.0f}W', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))

    # FIXED: Create proper grouped x-axis labels
    x_labels = []
    for i, batch_size in enumerate(batch_sizes):
        # ONE batch size label for the entire group
        label = f"{batch_size}"
        x_labels.append(label)
        
        # Add device labels under their respective bars (as part of x-axis)
        cpu_pos = x[i] + 0 * width
        mps_pos = x[i] + 1 * width
        
        ax.text(cpu_pos, -y_max*0.01, "device='cpu'", 
                ha='center', va='top', fontsize=10, fontweight='bold', color='blue')
        ax.text(mps_pos, -y_max*0.01, "device='mps'", 
                ha='center', va='top', fontsize=10, fontweight='bold', color='purple')
    
    # Set main x-axis labels (batch sizes centered under each group)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(x_labels, fontsize=11, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Batch Size Configurations', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power Consumption (Watts)', fontsize=12, fontweight='bold')
    ax.set_title('Power Composition vs Batch Size\n(Model: 4 layers, 4 heads, 256 embd, 128 block)', 
                 fontsize=14, fontweight='bold')
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cpu_color, alpha=0.8, label='CPU Power'),
        Patch(facecolor=gpu_color, alpha=0.8, label='GPU Power'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', 
              frameon=True, fancybox=True, shadow=True)
    
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

def plot_power_vs_block_size_enhanced(df, save_path=None):
    """
    Enhanced Plot 1B: Power composition across block sizes with proper grouped labeling
    """
    # Focus on one representative model size
    plot_data = df[
        (df['n_layer'] == 4) & 
        (df['n_head'] == 4) & 
        (df['n_embd'] == 256) &
        (df['batch_size'] == 32)
    ].copy()
    
    fig, ax = plt.subplots(figsize=(26, 9))
    
    # Get batch sizes and devices
    block_sizes = sorted(plot_data['block_size'].unique())
    devices = ['cpu', 'mps']
    
    # Colors - Red for CPU, Green for GPU
    cpu_color = '#d62728'  # Red
    gpu_color = '#2ca02c'  # Green
    
    # Bar settings
    x = np.arange(len(block_sizes))
    width = 0.35
    
    # Arrow configuration
    arrow_offset = 0.25
    arrow_props = dict(arrowstyle='->', color='gray', lw=1.5, alpha=0.7)
    
    # Calculate y_max BEFORE using it
    y_max = max(plot_data['avg_cpu_power'] + plot_data['avg_gpu_power']) * 1.3
    
    for i, device in enumerate(devices):
        device_data = plot_data[plot_data['device'] == device].sort_values('block_size')
        
        if not device_data.empty:
            cpu_power = device_data['avg_cpu_power'].values
            gpu_power = device_data['avg_gpu_power'].values
            
            x_pos = x + i * width
            
            # Stacked bars
            bars_cpu = ax.bar(x_pos, cpu_power, width, 
                             color=cpu_color, alpha=0.8)
            bars_gpu = ax.bar(x_pos, gpu_power, width, bottom=cpu_power,
                             color=gpu_color, alpha=0.8)
            
            # Add power value labels on the bars
            for j, (cpu, gpu) in enumerate(zip(cpu_power, gpu_power)):
                # CPU power label (centered in CPU section)
                if cpu > 10:
                    ax.text(x_pos[j], cpu/2, f'CPU: {cpu:.0f}W', 
                           ha='center', va='center', fontsize=8, fontweight='bold',
                           color='white')
                elif cpu > 0:
                    ax.annotate(f'CPU: {cpu:.0f}W', 
                               xy=(x_pos[j], cpu/2),
                               xytext=(x_pos[j] - arrow_offset, cpu/2 + 50),
                               arrowprops=arrow_props,
                               fontsize=7, fontweight='bold', ha='right')
                
                # GPU power label (centered in GPU section)
                gpu_center = cpu + gpu/2
                if gpu > 100:
                    ax.text(x_pos[j], gpu_center, f'GPU: {gpu:.0f}W', 
                           ha='center', va='center', fontsize=8, fontweight='bold',
                           color='white')
                elif gpu > 0:
                    ax.annotate(f'GPU: {gpu:.2f}W', 
                               xy=(x_pos[j], gpu_center),
                               xytext=(x_pos[j] - arrow_offset, gpu_center + 50),
                               arrowprops=arrow_props,
                               fontsize=7, fontweight='bold', ha='right')
                
                # Total power label at top
                total_power = cpu + gpu
                ax.text(x_pos[j], total_power + 15, f'Total: {total_power:.0f}W', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))

    # FIXED: Create proper grouped x-axis labels
    x_labels = []
    for i, block_size in enumerate(block_sizes):
        # ONE batch size label for the entire group
        label = f"{block_size}"
        x_labels.append(label)
        
        # Add device labels under their respective bars (as part of x-axis)
        cpu_pos = x[i] + 0 * width
        mps_pos = x[i] + 1 * width
        
        ax.text(cpu_pos, -y_max*0.01, "device='cpu'", 
                ha='center', va='top', fontsize=10, fontweight='bold', color='blue')
        ax.text(mps_pos, -y_max*0.01, "device='mps'", 
                ha='center', va='top', fontsize=10, fontweight='bold', color='purple')
    
    # Set main x-axis labels (batch sizes centered under each group)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(x_labels, fontsize=11, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Block Size Configurations', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power Consumption (Watts)', fontsize=12, fontweight='bold')
    ax.set_title('Power Composition vs Block Size\n(Model: 4 layers, 4 heads, 256 embd, 4 batch)', 
                 fontsize=14, fontweight='bold')
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cpu_color, alpha=0.8, label='CPU Power'),
        Patch(facecolor=gpu_color, alpha=0.8, label='GPU Power'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', 
              frameon=True, fancybox=True, shadow=True)
    
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

def plot_power_vs_model_architecture_enhanced(df, save_path=None):
    """
    Enhanced Plot 1C: Power composition across model architectures with proper grouped labeling
    """
    # Focus on one representative batch size
    plot_data = df[
        (df['batch_size'] == 2) & 
        (df['block_size'] == 128)
    ].copy()
    
    # Create model architecture labels
    def get_model_label(row):
        params_m = row['param_count_m']
        return f"L{row['n_layer']}H{row['n_head']}E{row['n_embd']}"
    
    plot_data['model_label'] = plot_data.apply(get_model_label, axis=1)
    
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Get unique model architectures sorted by parameter count
    model_archs = plot_data['model_label'].unique()
    
    devices = ['cpu', 'mps']
    
    # Colors - Red for CPU, Green for GPU
    cpu_color = '#d62728'  # Red
    gpu_color = '#2ca02c'  # Green
    
    # Bar settings
    x = np.arange(len(model_archs))
    width = 0.35
    
    # Arrow configuration
    arrow_offset = 0.25
    arrow_props = dict(arrowstyle='->', color='gray', lw=1.5, alpha=0.7)
    
    # Calculate y_max BEFORE using it
    y_max = max((plot_data['avg_cpu_power'] + plot_data['avg_gpu_power']).fillna(0)) * 1.3
    
    for i, device in enumerate(devices):
        device_data = plot_data[plot_data['device'] == device]
        
        # Ensure correct order
        device_data = device_data.set_index('model_label').reindex(model_archs).reset_index()
        
        cpu_power = device_data['avg_cpu_power'].fillna(0).values
        gpu_power = device_data['avg_gpu_power'].fillna(0).values
        
        x_pos = x + i * width
        
        # Stacked bars
        bars_cpu = ax.bar(x_pos, cpu_power, width, color=cpu_color, alpha=0.8)
        bars_gpu = ax.bar(x_pos, gpu_power, width, bottom=cpu_power, color=gpu_color, alpha=0.8)
        
        # Add power value labels on the bars
        for j, (cpu, gpu) in enumerate(zip(cpu_power, gpu_power)):
            # CPU power label
            if cpu > 10:
                ax.text(x_pos[j], cpu/2, f'CPU: {cpu:.0f}W', 
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       color='white')
            elif cpu > 0:
                ax.annotate(f'CPU: {cpu:.0f}W', 
                           xy=(x_pos[j], cpu/2), 
                           xytext=(x_pos[j] - arrow_offset, cpu/2 + 100),
                           arrowprops=arrow_props,
                           fontsize=7, fontweight='bold', ha='right')
            
            # GPU power label
            gpu_center = cpu + gpu/2
            if gpu > 50:
                ax.text(x_pos[j], gpu_center, f'GPU: {gpu:.0f}W', 
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       color='white')
            elif gpu > 0:
                ax.annotate(f'GPU: {gpu:.2f}W', 
                           xy=(x_pos[j], gpu_center), 
                           xytext=(x_pos[j] - arrow_offset, gpu_center + 100),
                           arrowprops=arrow_props,
                           fontsize=7, fontweight='bold', ha='right')
            
            # Total power label at top
            total_power = cpu + gpu
            if total_power > 0:
                ax.text(x_pos[j], total_power + 20, f'Total: {total_power:.0f}W', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))

    # FIXED: Create proper grouped x-axis labels
    x_labels = []
    for i, arch in enumerate(model_archs):
        # ONE model architecture label for the entire group
        x_labels.append(arch)
        
        # Add device labels under their respective bars
        cpu_pos = x[i] + 0 * width
        mps_pos = x[i] + 1 * width
        
        ax.text(cpu_pos, -y_max*0.01, "device='cpu'", 
                ha='center', va='top', fontsize=9, fontweight='bold', color='blue')
        ax.text(mps_pos, -y_max*0.01, "device='mps'", 
                ha='center', va='top', fontsize=9, fontweight='bold', color='purple')
    
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(x_labels, fontsize=10)
        
    # Customize plot
    ax.set_xlabel('Model Architecture Configurations', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power Consumption (Watts)', fontsize=12, fontweight='bold')
    ax.set_title('Power Composition vs Model Architecture\n(Batch Size: 2, Block Size: 128)', 
                 fontsize=14, fontweight='bold')
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cpu_color, alpha=0.8, label='CPU Power'),
        Patch(facecolor=gpu_color, alpha=0.8, label='GPU Power'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', 
              frameon=True, fancybox=True, shadow=True)
    
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
    
fig_1a = plot_power_vs_batch_size_enhanced(df, save_path='results/plot_1a_power_vs_batch_size.png')
fig_1b = plot_power_vs_block_size_enhanced(df, save_path='results/plot_1b_power_vs_block_size.png')
fig_1c = plot_power_vs_model_architecture_enhanced(df, save_path='results/plot_1c_power_vs_model_arch.png')