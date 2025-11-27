import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, root_dir)
import utils.plot_helper as plot_helper
load_latest_results = plot_helper.load_latest_results
df, latest_file = load_latest_results()

def plot_dual_axis_steps_power_vs_batch(df, save_path=None):
    """
    Plot 8A: Dual-axis - Steps/sec and Power vs Batch size
    """
    # Focus on one representative model size and block size
    plot_data = df[
        (df['n_layer'] == 4) & 
        (df['n_head'] == 4) & 
        (df['n_embd'] == 256) &
        (df['block_size'] == 64)
    ].sort_values(['device', 'batch_size'])
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    devices = ['cpu', 'mps']
    colors_steps = ['#1f77b4', '#ff7f0e']  # Blue for CPU, Orange for MPS
    colors_power = ['#2ca02c', '#d62728']  # Green for CPU, Red for MPS
    
    # Create second y-axis
    ax2 = ax1.twinx()
    
    for i, device in enumerate(devices):
        device_data = plot_data[plot_data['device'] == device]
        
        if not device_data.empty:
            # Plot steps/second on left axis
            line1 = ax1.plot(device_data['batch_size'], device_data['steps_per_second'],
                           marker='o', linewidth=3, markersize=8,
                           color=colors_steps[i], label=f'{device.upper()} Throughput')
            
            # Plot power on right axis
            line2 = ax2.plot(device_data['batch_size'], device_data['total_power'],
                           marker='s', linewidth=3, markersize=8,
                           color=colors_power[i], label=f'{device.upper()} Power')
            
            # Add value annotations
            for j, (batch, steps, power) in enumerate(zip(device_data['batch_size'], 
                                                        device_data['steps_per_second'], 
                                                        device_data['total_power'])):
                ax1.annotate(f'{steps:.0f}', 
                           xy=(batch, steps), 
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', va='bottom', fontsize=9, fontweight='bold',
                           color=colors_steps[i])
                
                ax2.annotate(f'{power:.0f}W', 
                           xy=(batch, power), 
                           xytext=(0, -15), textcoords='offset points',
                           ha='center', va='top', fontsize=9, fontweight='bold',
                           color=colors_power[i])
    
    # Customize axes
    ax1.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Throughput (steps/second)', fontsize=12, fontweight='bold', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_ylabel('Total Power (Watts)', fontsize=12, fontweight='bold', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax1.set_title('Throughput vs Power: Dual-Axis Analysis\n(Model: 4 layers, 4 heads, 256 embds, 64 block)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dual-axis steps/power plot saved to {save_path}")
    
    return fig

def plot_radar_charts_by_architecture(df, save_path=None):
    """
    Plot 8B: Three radar charts comparing best configurations for each model architecture
    """
    # Define the three model architectures we want to analyze
    architectures = [
        {'name': 'Small Model', 'n_layer': 2, 'n_head': 2, 'n_embd': 128},
        {'name': 'Medium Model', 'n_layer': 4, 'n_head': 4, 'n_embd': 256},
        {'name': 'Large Model', 'n_layer': 6, 'n_head': 6, 'n_embd': 384}
    ]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(26, 10), subplot_kw=dict(projection='polar'))
    
    # Colors for different configurations
    config_colors = {
        'Best Throughput': '#1f77b4',
        'Best Efficiency': '#ff7f0e', 
        'Power Efficient': '#2ca02c',
        'Lowest Power': '#2ca02c',
        'Best Loss': '#d62728'
    }
    
    # Metrics for radar chart
    metrics = ['steps_per_second', 'flops_per_watt', 'total_power', 'final_loss', 'training_time']
    metric_labels = ['Throughput\n(steps/s)', 'Efficiency\n(FLOPs/W)', 'Power\nConsumption', 'Loss\nQuality', 'Training\nSpeed']
    
    # Calculate angles for radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for arch_idx, arch in enumerate(architectures):
        ax = axes[arch_idx]
        
        # Filter data for this specific architecture AND successful runs only
        arch_data = df[
            (df['n_layer'] == arch['n_layer']) &
            (df['n_head'] == arch['n_head']) &
            (df['n_embd'] == arch['n_embd']) &
            (df['final_loss'] != 0)  # Only successful runs
        ].copy()
        
        if arch_data.empty:
            ax.text(0.5, 0.5, f"No successful runs\nfor {arch['name']}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, fontweight='bold')
            ax.set_title(f"{arch['name']}\nL{arch['n_layer']}H{arch['n_head']}E{arch['n_embd']}\n(No successful runs)", 
                        fontsize=12, fontweight='bold', pad=20)
            continue
        
        print(f"\n{arch['name']} (L{arch['n_layer']}H{arch['n_head']}E{arch['n_embd']}): {len(arch_data)} successful configurations")
        
        # Select best configurations for this architecture - ENSURING UNIQUE CONFIGS
        best_configs = []
        used_indices = set()
        
        # 1. Best throughput
        best_throughput_idx = arch_data['steps_per_second'].idxmax()
        best_configs.append(('Best Throughput', arch_data.loc[best_throughput_idx]))
        used_indices.add(best_throughput_idx)
        print(f"  - Best Throughput: {arch_data.loc[best_throughput_idx]['steps_per_second']:.0f} steps/s")
        
        # 2. Best efficiency  
        # Exclude the throughput config if it's the same
        efficiency_candidates = arch_data[~arch_data.index.isin(used_indices)]
        if not efficiency_candidates.empty:
            best_efficiency_idx = efficiency_candidates['flops_per_watt'].idxmax()
        else:
            best_efficiency_idx = arch_data['flops_per_watt'].idxmax()
        best_configs.append(('Best Efficiency', arch_data.loc[best_efficiency_idx]))
        used_indices.add(best_efficiency_idx)
        print(f"  - Best Efficiency: {arch_data.loc[best_efficiency_idx]['flops_per_watt']:.0f} FLOPs/W")
        
        # 3. Power Efficient - ENSURING DIFFERENT CONFIG
        # Find configurations in the lower 50% of power consumption but with good efficiency
        power_threshold = arch_data['total_power'].quantile(0.5)
        power_efficient_candidates = arch_data[
            (arch_data['total_power'] <= power_threshold) & 
            (~arch_data.index.isin(used_indices))  # Exclude already selected configs
        ]
        
        if not power_efficient_candidates.empty:
            # Among low-power configs, pick the one with best efficiency
            best_power_idx = power_efficient_candidates['flops_per_watt'].idxmax()
            best_configs.append(('Power Efficient', arch_data.loc[best_power_idx]))
            used_indices.add(best_power_idx)
            print(f"  - Power Efficient: {arch_data.loc[best_power_idx]['total_power']:.0f}W, {arch_data.loc[best_power_idx]['flops_per_watt']:.0f} FLOPs/W")
        else:
            # If no unique low-power candidates, find any low power config
            low_power_candidates = arch_data[arch_data['total_power'] <= power_threshold]
            if not low_power_candidates.empty:
                best_power_idx = low_power_candidates['flops_per_watt'].idxmax()
                best_configs.append(('Power Efficient', arch_data.loc[best_power_idx]))
                used_indices.add(best_power_idx)
                print(f"  - Power Efficient (non-unique): {arch_data.loc[best_power_idx]['total_power']:.0f}W, {arch_data.loc[best_power_idx]['flops_per_watt']:.0f} FLOPs/W")
            else:
                # Fallback: absolute lowest power
                lowest_power_candidates = arch_data[~arch_data.index.isin(used_indices)]
                if not lowest_power_candidates.empty:
                    best_power_idx = lowest_power_candidates['total_power'].idxmin()
                else:
                    best_power_idx = arch_data['total_power'].idxmin()
                best_configs.append(('Lowest Power', arch_data.loc[best_power_idx]))
                used_indices.add(best_power_idx)
                print(f"  - Lowest Power: {arch_data.loc[best_power_idx]['total_power']:.0f}W")
        
        # 4. Best loss - ENSURING DIFFERENT CONFIG
        loss_candidates = arch_data[~arch_data.index.isin(used_indices)]
        if not loss_candidates.empty:
            best_loss_idx = loss_candidates['final_loss'].idxmin()
        else:
            best_loss_idx = arch_data['final_loss'].idxmin()
        best_configs.append(('Best Loss', arch_data.loc[best_loss_idx]))
        used_indices.add(best_loss_idx)
        print(f"  - Best Loss: {arch_data.loc[best_loss_idx]['final_loss']:.3f}")
        
        # Create detailed legend labels with configuration info
        legend_labels = []
        for config_name, config in best_configs:
            device = 'MPS' if config['device'] == 'mps' else 'CPU'
            batch_block = f"B{config['batch_size']}x{config['block_size']}"
            legend_labels.append(f"{config_name}\n{batch_block} ({device})")
        
        print(f"  Selected {len(best_configs)} unique configurations:")
        for label in legend_labels:
            print(f"    {label.replace(chr(10), ' - ')}")
        
        # Calculate min and max for normalization (for this architecture only)
        metric_ranges = {}
        for metric in metrics:
            if metric in ['total_power', 'final_loss', 'training_time']:
                metric_ranges[metric] = {
                    'min': arch_data[metric].min(),
                    'max': arch_data[metric].max(),
                    'invert': True
                }
            else:
                metric_ranges[metric] = {
                    'min': arch_data[metric].min(),
                    'max': arch_data[metric].max(),
                    'invert': False
                }
        
        # Normalize data and plot for this architecture
        for i, (config_name, config) in enumerate(best_configs):
            color = config_colors[config_name]
            
            normalized_values = []
            for metric in metrics:
                min_val = metric_ranges[metric]['min']
                max_val = metric_ranges[metric]['max']
                current_val = config[metric]
                
                if max_val == min_val:
                    normalized = 1.0
                else:
                    if metric_ranges[metric]['invert']:
                        normalized = 1.0 - ((current_val - min_val) / (max_val - min_val))
                    else:
                        normalized = (current_val - min_val) / (max_val - min_val)
                
                normalized = max(0.0, min(1.0, normalized))
                normalized_values.append(normalized)
            
            # Complete the circle
            values = normalized_values + [normalized_values[0]]
            
            # Plot the line
            ax.plot(angles, values, 'o-', linewidth=2.5, markersize=8, 
                   color=color, label=legend_labels[i], markeredgecolor='black', markeredgewidth=0.5)
                    
        # Customize the radar chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=10)#, fontweight='bold')
        ax.tick_params(axis='x', pad=20)
        # Set radial labels
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add radial grid lines
        ax.set_rlabel_position(30)
        
        # Add legend with detailed configuration info
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=8, 
                 frameon=True, fancybox=True, shadow=True,
                 title="Configuration Details", title_fontsize=9)
        
        # Add architecture details to title
        device_counts = arch_data['device'].value_counts()
        device_info = f"CPU: {device_counts.get('cpu', 0)}, MPS: {device_counts.get('mps', 0)}"
        
        ax.set_title(f"{arch['name']}\n"
                    f"L{arch['n_layer']}H{arch['n_head']}E{arch['n_embd']}\n"
                    f"{device_info}", 
                    fontsize=12, fontweight='bold', pad=25)
    
    # Add overall title
    #plt.suptitle('Radar Charts: Best Configurations by Model Architecture\n(Only Successful Runs - Loss ≠ 0)', 
    #            fontsize=16, fontweight='bold', y=0.95)
    
    # Add global explanation
    explanation = "Metrics normalized per architecture\n" \
                 "• Throughput: Higher steps/second = Better\n" \
                 "• Efficiency: Higher FLOPs/Watt = Better\n" \
                 "• Power: Lower consumption = Better\n" \
                 "• Loss: Lower value = Better\n" \
                 "• Training: Faster time = Better"
    
    fig.text(0.02, 0.02, explanation, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8),
            verticalalignment='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Architecture radar charts saved to {save_path}")

    return fig

def plot_scatter_matrix_correlations(df, save_path=None):
    """
    Plot 8C: Scatter matrix - Correlations between all major metrics
    """
    # Select key metrics for correlation analysis
    metrics = ['steps_per_second', 'flops_per_watt', 'total_power', 'final_loss', 
              'training_time', 'param_count_m', 'batch_size', 'block_size']
    
    metric_names = {
        'steps_per_second': 'Throughput\n(steps/s)',
        'flops_per_watt': 'Efficiency\n(FLOPs/W)',
        'total_power': 'Total Power\n(W)',
        'final_loss': 'Final Loss',
        'training_time': 'Training Time\n(s)',
        'param_count_m': 'Model Size\n(M params)',
        'batch_size': 'Batch Size',
        'block_size': 'Block Size'
    }
    
    # Create a clean copy and filter out invalid data
    clean_df = df.copy()
    
    # Filter out inf, nan, and invalid values for each metric
    for metric in metrics:
        if metric in ['final_loss', 'steps_per_second', 'flops_per_watt', 'training_time', 'total_power']:
            clean_df = clean_df[
                (clean_df[metric] > 0) & 
                (clean_df[metric] < np.inf) &
                (clean_df[metric].notna())
            ]
    
    if clean_df.empty:
        print("No valid data available for scatter matrix")
        return None
    
    plot_data = clean_df[metrics].copy()
    
    print(f"Using {len(clean_df)} valid configurations for scatter matrix")
    
    # Create scatter matrix
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(len(metrics), len(metrics))
    
    devices = clean_df['device'].unique()
    colors = {'cpu': '#1f77b4', 'mps': '#ff7f0e'}
    
    for i, metric_x in enumerate(metrics):
        for j, metric_y in enumerate(metrics):
            ax = fig.add_subplot(gs[i, j])
            
            if i == j:
                # Diagonal - histograms (using clean data)
                for device in devices:
                    device_data = clean_df[clean_df['device'] == device]
                    # Filter out any remaining invalid values for this specific metric
                    valid_data = device_data[metric_x][
                        (device_data[metric_x] > 0) & 
                        (device_data[metric_x] < np.inf) &
                        (device_data[metric_x].notna())
                    ]
                    if len(valid_data) > 0:
                        ax.hist(valid_data, alpha=0.7, color=colors[device], 
                               label=device.upper(), bins=15)
                
                ax.set_title(f'Distribution: {metric_names[metric_x]}', fontsize=10, fontweight='bold')
                if i == len(metrics)-1:
                    ax.legend(fontsize=8)
            else:
                # Off-diagonal - scatter plots (using clean data)
                for device in devices:
                    device_data = clean_df[clean_df['device'] == device]
                    # Filter for valid pairs
                    valid_mask = (
                        (device_data[metric_x] > 0) & (device_data[metric_x] < np.inf) &
                        (device_data[metric_x].notna()) &
                        (device_data[metric_y] > 0) & (device_data[metric_y] < np.inf) &
                        (device_data[metric_y].notna())
                    )
                    valid_data = device_data[valid_mask]
                    
                    if len(valid_data) > 0:
                        ax.scatter(valid_data[metric_x], valid_data[metric_y], 
                                  alpha=0.6, color=colors[device], label=device.upper(), s=50)
                
                # Calculate and display correlation using clean data
                valid_corr_data = plot_data[
                    (plot_data[metric_x] > 0) & (plot_data[metric_x] < np.inf) &
                    (plot_data[metric_x].notna()) &
                    (plot_data[metric_y] > 0) & (plot_data[metric_y] < np.inf) &
                    (plot_data[metric_y].notna())
                ]
                
                if len(valid_corr_data) >= 2:  # Need at least 2 points for correlation
                    corr = valid_corr_data[metric_x].corr(valid_corr_data[metric_y])
                    ax.text(0.05, 0.95, f'ρ = {corr:.2f}', transform=ax.transAxes,
                           fontsize=10, fontweight='bold', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
                else:
                    ax.text(0.05, 0.95, 'ρ = N/A', transform=ax.transAxes,
                           fontsize=10, fontweight='bold', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
            
            # Set labels only on edges
            if i == len(metrics)-1:
                ax.set_xlabel(metric_names[metric_x], fontsize=10, fontweight='bold')
            if j == 0:
                ax.set_ylabel(metric_names[metric_y], fontsize=10, fontweight='bold')
            
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
    
    # Add filtering info to title
    original_count = len(df)
    valid_count = len(clean_df)
    filtered_info = ""
    if original_count > valid_count:
        filtered_info = f" ({original_count - valid_count} invalid runs filtered)"
    
    plt.suptitle(f'Scatter Matrix: Correlations Between All Major Metrics{filtered_info}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter matrix saved to {save_path}")
        print(f"Filtered out {original_count - valid_count} invalid configurations")
    
    return fig

fig_8a = plot_dual_axis_steps_power_vs_batch(df, save_path='results/plot_8a_dual_axis_steps_power.png')
fig_8b= plot_radar_charts_by_architecture(df, save_path='results/plot_8b_radar_combined.png')
fig_8c = plot_scatter_matrix_correlations(df, save_path='results/plot_8c_scatter_matrix.png')