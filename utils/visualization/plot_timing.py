import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, root_dir)
import utils.plot_helper as plot_helper
load_latest_results = plot_helper.load_latest_results
df, latest_file = load_latest_results()

def create_smart_pie_chart(ax, sizes, labels, colors, threshold=10):
    """
    Create a pie chart with smart label placement:
    - Labels > threshold%: inside the pie
    - Labels <= threshold%: outside with arrows
    """
    # Create pie chart without labels initially
    wedges, texts, autotexts = ax.pie(
        sizes, colors=colors, 
        autopct=lambda pct: f'{pct:.1f}%' if pct >= threshold else '',
        startangle=90, shadow=True,
        labels=None,  # No labels on slices
        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
    )
    
    # Store wedge information for external labels
    external_labels = []
    
    # Process each wedge
    for i, (wedge, pct, label) in enumerate(zip(wedges, sizes, labels)):
        angle = (wedge.theta2 + wedge.theta1) / 2
        radius = wedge.r
        if pct < threshold:
            # For small slices, place all text on the right side with horizontal arrows
            # Stagger vertical positions to avoid overlap
            base_y = 0
            vertical_spacing = 0.15
            text_y = base_y + (len(external_labels) * vertical_spacing)
            
            x = 1  # Fixed x position on the right
            y = text_y+1
            
            # Add external text on the right side
            text = ax.text(x, y, f'{pct:.1f}%', 
                        ha='left', va='center',
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Calculate arrow endpoint to point to the actual wedge
            arrow_x = radius * 0.9 * np.cos(np.radians(angle))
            arrow_y = radius * 0.9 * np.sin(np.radians(angle))
            
            # Arrow starts from near the text (slightly left of text position)
            arrow_start_x = x - 0.1
            arrow_start_y = y
            
            ax.annotate('', xy=(arrow_x, arrow_y), xytext=(arrow_start_x, arrow_start_y),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5, alpha=0.8),
                    zorder=10)
            
            external_labels.append((wedge, label, pct))
        else:
            # For larger slices, enhance the existing autotext
            if autotexts[i].get_text():  # Only if we have text
                autotexts[i].set_color('white')
                autotexts[i].set_fontweight('bold')
                autotexts[i].set_fontsize(10)
    
    return wedges, external_labels

def plot_component_timing_breakdown(df, save_path=None):
    """
    Plot 4: Component Timing Breakdown
    Detailed analysis of forward/backward/optimizer timing distributions using actual measurements
    """
    # Focus on representative configurations
    representative_configs = [
        {'n_layer': 2, 'n_head': 2, 'n_embd': 128, 'batch_size': 8, 'block_size': 64, 'device': 'mps'},
        {'n_layer': 4, 'n_head': 4, 'n_embd': 256, 'batch_size': 8, 'block_size': 64, 'device': 'mps'},
        {'n_layer': 4, 'n_head': 4, 'n_embd': 256, 'batch_size': 8, 'block_size': 64, 'device': 'cpu'},
    ]
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create subplot grid
    gs = fig.add_gridspec(2, 3)
    
    # Plot 1: Pie charts for component timing distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Plot 2: Stacked area chart of component timing vs batch size
    ax4 = fig.add_subplot(gs[1, :])
    
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']  # Forward, Backward, Optimizer, Other
    
    # Plot pie charts for each representative configuration
    pie_axes = [ax1, ax2, ax3]
    for i, config in enumerate(representative_configs):
        config_data = df[
            (df['n_layer'] == config['n_layer']) &
            (df['n_head'] == config['n_head']) &
            (df['n_embd'] == config['n_embd']) &
            (df['batch_size'] == config['batch_size']) &
            (df['block_size'] == config['block_size']) &
            (df['device'] == config['device'])
        ]
        
        if not config_data.empty:
            row = config_data.iloc[0]
            
            # Use actual timing measurements
            forward_time = row['forward_total_time_s']
            backward_time = row['backward_total_time_s']
            optimizer_time = row['optimizer_step_time_s'] + row['zero_grad_time_s']
            
            # Calculate other components
            other_time = row['training_time'] - (forward_time + backward_time + optimizer_time)
            
            times = [forward_time, backward_time, optimizer_time, other_time]
            labels = ['Forward', 'Backward', 'Optimizer', 'Other']
            
            wedges, texts, autotexts = pie_axes[i].pie(
                times, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, shadow=True
            )
            
            # Enhance pie chart appearance
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            device_label = 'MPS' if config['device'] == 'mps' else 'CPU'
            config_label = f"L{config['n_layer']}H{config['n_head']}E{config['n_embd']}\nB{config['batch_size']}x{config['block_size']} ({device_label})"
            pie_axes[i].set_title(f'Timing Distribution\n{config_label}', fontweight='bold')
    
    # Plot stacked area chart: Component timing vs batch size for MPS device
    batch_sizes = sorted(df[(df['device'] == 'mps') & 
                           (df['n_layer'] == 4) & 
                           (df['n_head'] == 4) & 
                           (df['n_embd'] == 256)]['batch_size'].unique())
    
    forward_times = []
    backward_times = []
    optimizer_times = []
    other_times = []
    
    for batch_size in batch_sizes:
        batch_data = df[
            (df['device'] == 'mps') &
            (df['n_layer'] == 4) &
            (df['n_head'] == 4) &
            (df['n_embd'] == 256) &
            (df['batch_size'] == batch_size) &
            (df['block_size'] == 128)
        ]
        
        if not batch_data.empty:
            row = batch_data.iloc[0]
            forward_times.append(row['forward_total_time_s'])
            backward_times.append(row['backward_total_time_s'])
            optimizer_times.append(row['optimizer_step_time_s'] + row['zero_grad_time_s'])
            other_times.append(row['training_time'] - (row['forward_total_time_s'] + 
                                                     row['backward_total_time_s'] + 
                                                     row['optimizer_step_time_s'] + 
                                                     row['zero_grad_time_s']))
    
    # Create stacked area chart
    if forward_times and backward_times and optimizer_times and other_times:
        ax4.stackplot(batch_sizes, forward_times, backward_times, optimizer_times, other_times,
                     labels=['Forward', 'Backward', 'Optimizer', 'Other'],
                     colors=colors, alpha=0.8)
        
        ax4.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Time per Component (seconds)', fontsize=12, fontweight='bold')
        ax4.set_title('Component Timing vs Batch Size\n(Model: 4L-4H-256E, Block: 128, Device: MPS)', 
                     fontsize=14, fontweight='bold')
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Component timing breakdown plot saved to {save_path}")

    return fig

def plot_transformer_block_timing_breakdown(df, save_path=None):
    """
    Plot 7A: Attention vs MLP timing analysis with smart pie chart labels
    """
    # Create subplots for pie charts
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Model configurations to analyze
    model_configs = [
        {'n_layer': 2, 'n_head': 2, 'n_embd': 128, 'batch_size': 8, 'block_size': 64, 'label': 'Small\nL2H2E128'},
        {'n_layer': 4, 'n_head': 4, 'n_embd': 256, 'batch_size': 8, 'block_size': 64, 'label': 'Medium\nL4H4E256'},
        {'n_layer': 6, 'n_head': 6, 'n_embd': 384, 'batch_size': 8, 'block_size': 64, 'label': 'Large\nL6H6E384'},
    ]
    
    colors = ['#ff6b6b', '#4ecdc4', '#ffe66d', '#95e1d3', '#ff9ff3']  # Attention, MLP, LayerNorm, Residual, Other
    
    # Plot pie charts for each configuration
    for i, config in enumerate(model_configs):
        config_data = df[
            (df['n_layer'] == config['n_layer']) &
            (df['n_head'] == config['n_head']) &
            (df['n_embd'] == config['n_embd']) &
            (df['batch_size'] == config['batch_size']) &
            (df['block_size'] == config['block_size']) &
            (df['device'] == 'mps')
        ]
        
        if not config_data.empty:
            row = config_data.iloc[0]
            total_block_time = row['block_total_time_s']
            
            # Extract component times
            attention_time = row['block_attention_time_s']
            mlp_time = row['block_mlp_time_s']
            layernorm_time = row['block_layernorm1_time_s'] + row['block_layernorm2_time_s']
            residual_time = row['block_residual1_time_s'] + row['block_residual2_time_s']
            
            # Calculate other time
            other_time = max(0, total_block_time - (attention_time + mlp_time + layernorm_time + residual_time))
            
            times = [attention_time, mlp_time, layernorm_time, residual_time, other_time]
            percentages = [t / total_block_time * 100 for t in times]
            labels = ['Attention', 'MLP', 'Layer Norm', 'Residual', 'Other']
            
            # Filter out zero components for cleaner pie chart
            non_zero_times = []
            non_zero_percentages = []
            non_zero_labels = []
            non_zero_colors = []
            for j, (time, pct) in enumerate(zip(times, percentages)):
                if time > 0.001:  # Only include components with significant time
                    non_zero_times.append(time)
                    non_zero_percentages.append(pct)
                    non_zero_labels.append(labels[j])
                    non_zero_colors.append(colors[j])
            
            # Create smart pie chart
            wedges, external_labels = create_smart_pie_chart(
                axes[i], non_zero_percentages, non_zero_labels, non_zero_colors, threshold=10
            )
            
            # Add legend for this subplot
            axes[i].legend(wedges, non_zero_labels, 
                          title="Components",
                          loc="center left",
                          bbox_to_anchor=(1, 0, 0.5, 1),
                          fontsize=9)
            
            # Add parameter count to title
            params = row['param_count_m']
            axes[i].set_title(f'{config["label"]}\n{params:.1f}M params', fontweight='bold', fontsize=12)
        else:
            axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[i].transAxes, fontsize=14)
            axes[i].set_title(config['label'], fontweight='bold', fontsize=12)
    
    fig.suptitle('Transformer Block Component Timing Breakdown\n(MPS Device, Batch: 8, Block: 64)', 
                fontsize=16, fontweight='normal', y=0.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention vs MLP analysis plot saved to {save_path}")

    return fig

def plot_attention_timing_breakdown(df, save_path=None):
    """
    Plot 7B: Forward/Backward ratio analysis with smart pie chart labels
    """
    fig = plt.figure(figsize=(22, 12))
    
    # Create subplot grid: 2 rows, 3 columns
    gs = fig.add_gridspec(2, 3)
    
    # Top row: Model size comparison (CPU vs MPS)
    ax1 = fig.add_subplot(gs[0, 0])  # Small model - CPU
    ax2 = fig.add_subplot(gs[0, 1])  # Medium model - CPU  
    ax3 = fig.add_subplot(gs[0, 2])  # Large model - CPU
    
    ax4 = fig.add_subplot(gs[1, 0])  # Small model - MPS
    ax5 = fig.add_subplot(gs[1, 1])  # Medium model - MPS
    ax6 = fig.add_subplot(gs[1, 2])  # Large model - MPS
    
    # Model configurations
    model_configs = [
        {'n_layer': 2, 'n_head': 2, 'n_embd': 128, 'batch_size': 8, 'block_size': 64, 'label': 'Small'},
        {'n_layer': 4, 'n_head': 4, 'n_embd': 256, 'batch_size': 8, 'block_size': 64, 'label': 'Medium'},
        {'n_layer': 6, 'n_head': 6, 'n_embd': 384, 'batch_size': 8, 'block_size': 64, 'label': 'Large'},
    ]
    
    devices = ['cpu', 'mps']
    colors = ['#2ca02c', '#d62728', '#ff7f0e', '#9467bd', '#8c564b',]  # Forward, Backward, Optimizer, Data, Other
    
    # Plot pie charts for each combination
    for i, config in enumerate(model_configs):
        for j, device in enumerate(devices):
            config_data = df[
                (df['n_layer'] == config['n_layer']) &
                (df['n_head'] == config['n_head']) &
                (df['n_embd'] == config['n_embd']) &
                (df['batch_size'] == config['batch_size']) &
                (df['block_size'] == config['block_size']) &
                (df['device'] == device)
            ]
            
            # Determine which axis to use
            if device == 'cpu':
                ax = [ax1, ax2, ax3][i]
            else:
                ax = [ax4, ax5, ax6][i]
            
            if not config_data.empty:
                row = config_data.iloc[0]
                total_time = row['attention_total_time_s']
                
                # Calculate component percentages
                forward_pct = (row['attention_qkv_proj_time_s'] / total_time) * 100
                backward_pct = (row['attention_reshape_time_s'] / total_time) * 100
                optimizer_pct = (row['attention_compute_time_s'] / total_time) * 100
                data_pct = (row['attention_output_reassemble_time_s'] / total_time) * 100
                other_pct = 100 - (forward_pct + backward_pct + optimizer_pct + data_pct)
                
                percentages = [forward_pct, optimizer_pct, backward_pct, data_pct, other_pct]
                labels = ['QKV projections', 'Computation', 'Reshaping', 'Output Reassembling', 'Other']
                
                # Filter out negligible components
                non_zero_percentages = []
                non_zero_labels = []
                non_zero_colors = []
                for k, (pct, label) in enumerate(zip(percentages, labels)):
                    if pct > 0.1:  # Only show components > 0.5%
                        non_zero_percentages.append(pct)
                        non_zero_labels.append(label)
                        non_zero_colors.append(colors[k])
                
                if non_zero_percentages:
                    # Create smart pie chart
                    wedges, external_labels = create_smart_pie_chart(
                        ax, non_zero_percentages, non_zero_labels, non_zero_colors, threshold=8
                    )
                    
                    # Add legend for this subplot
                    ax.legend(wedges, non_zero_labels, 
                             title="Components",
                             loc="center left",
                             bbox_to_anchor=(1, 0, 10, 1),
                             fontsize=8)
                
                device_label = 'CPU' if device == 'cpu' else 'MPS'
                params = row['param_count_m']
                ax.set_title(f'{config["label"]} Model ({device_label})\n{params:.1f}M params', 
                           fontweight='bold', fontsize=11)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
                device_label = 'CPU' if device == 'cpu' else 'MPS'
                ax.set_title(f'{config["label"]} Model ({device_label})', fontweight='bold', fontsize=11)
    
    fig.suptitle('Training Component Breakdown by Model Size and Device\n(Batch: 8, Block: 64)', 
                fontsize=16, fontweight='normal', y=0.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Forward/backward ratio analysis plot saved to {save_path}")

    return fig

def plot_mlp_timing_breakdown(df, save_path=None):
    """
    Plot 7C: Forward/Backward ratio analysis with smart pie chart labels
    """
    fig = plt.figure(figsize=(22, 12))
    
    # Create subplot grid: 2 rows, 3 columns
    gs = fig.add_gridspec(2, 3)
    
    # Top row: Model size comparison (CPU vs MPS)
    ax1 = fig.add_subplot(gs[0, 0])  # Small model - CPU
    ax2 = fig.add_subplot(gs[0, 1])  # Medium model - CPU  
    ax3 = fig.add_subplot(gs[0, 2])  # Large model - CPU
    
    ax4 = fig.add_subplot(gs[1, 0])  # Small model - MPS
    ax5 = fig.add_subplot(gs[1, 1])  # Medium model - MPS
    ax6 = fig.add_subplot(gs[1, 2])  # Large model - MPS
    
    # Model configurations
    model_configs = [
        {'n_layer': 2, 'n_head': 2, 'n_embd': 128, 'batch_size': 8, 'block_size': 64, 'label': 'Small'},
        {'n_layer': 4, 'n_head': 4, 'n_embd': 256, 'batch_size': 8, 'block_size': 64, 'label': 'Medium'},
        {'n_layer': 6, 'n_head': 6, 'n_embd': 384, 'batch_size': 8, 'block_size': 64, 'label': 'Large'},
    ]
    
    devices = ['cpu', 'mps']
    colors = ['#2ca02c', '#d62728', '#ff7f0e', '#9467bd', '#8c564b',]  # Forward, Backward, Optimizer, Data, Other
    
    # Plot pie charts for each combination
    for i, config in enumerate(model_configs):
        for j, device in enumerate(devices):
            config_data = df[
                (df['n_layer'] == config['n_layer']) &
                (df['n_head'] == config['n_head']) &
                (df['n_embd'] == config['n_embd']) &
                (df['batch_size'] == config['batch_size']) &
                (df['block_size'] == config['block_size']) &
                (df['device'] == device)
            ]
            
            # Determine which axis to use
            if device == 'cpu':
                ax = [ax1, ax2, ax3][i]
            else:
                ax = [ax4, ax5, ax6][i]
            
            if not config_data.empty:
                row = config_data.iloc[0]
                total_time = row['mlp_total_time_s']
                
                # Calculate component percentages
                forward_pct = (row['mlp_fc1_time_s'] / total_time) * 100
                backward_pct = (row['mlp_activation_time_s'] / total_time) * 100
                optimizer_pct = (row['mlp_fc2_time_s'] / total_time) * 100
                data_pct = (row['mlp_dropout_time_s'] / total_time) * 100
                other_pct = 100 - (forward_pct + backward_pct + optimizer_pct + data_pct)
                
                percentages = [forward_pct, optimizer_pct, backward_pct, data_pct, other_pct]
                labels = ['Up projection', 'Actication', 'Down projection', 'Dropout', 'Other']
                
                # Filter out negligible components
                non_zero_percentages = []
                non_zero_labels = []
                non_zero_colors = []
                for k, (pct, label) in enumerate(zip(percentages, labels)):
                    if pct > 0.1:  # Only show components > 0.5%
                        non_zero_percentages.append(pct)
                        non_zero_labels.append(label)
                        non_zero_colors.append(colors[k])
                
                if non_zero_percentages:
                    # Create smart pie chart
                    wedges, external_labels = create_smart_pie_chart(
                        ax, non_zero_percentages, non_zero_labels, non_zero_colors, threshold=8
                    )
                    
                    # Add legend for this subplot
                    ax.legend(wedges, non_zero_labels, 
                             title="Components",
                             loc="center left",
                             bbox_to_anchor=(1, 0, 10, 1),
                             fontsize=8)
                
                device_label = 'CPU' if device == 'cpu' else 'MPS'
                params = row['param_count_m']
                ax.set_title(f'{config["label"]} Model ({device_label})\n{params:.1f}M params', 
                           fontweight='bold', fontsize=11)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
                device_label = 'CPU' if device == 'cpu' else 'MPS'
                ax.set_title(f'{config["label"]} Model ({device_label})', fontweight='bold', fontsize=11)
    
    fig.suptitle('Training Component Breakdown by Model Size and Device\n(Batch: 8, Block: 64)', 
                fontsize=16, fontweight='normal', y=0.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Forward/backward ratio analysis plot saved to {save_path}")

    return fig

fig_7a = plot_transformer_block_timing_breakdown(df, save_path='results/plot_7a_transformer_block_timing.png')
fig_7b = plot_attention_timing_breakdown(df, save_path='results/plot_7b_attention_timing.png')
fig_7c = plot_mlp_timing_breakdown(df, save_path='results/plot_7c_mlp_timing.png')