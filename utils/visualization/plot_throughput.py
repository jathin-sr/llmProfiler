import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, root_dir)
import utils.plot_helper as plot_helper
load_latest_results = plot_helper.load_latest_results
df, latest_file = load_latest_results()

def plot_throughput_vs_power_by_model(df, save_path=None):
    """
    Plot 3A: Steps/second vs Total Power for each model architecture
    Separate plots for CPU and MPS devices
    """
    # Calculate total power
    df['total_power'] = df['avg_cpu_power'] + df['avg_gpu_power']
    
    # Create model architecture labels WITH parameter count to match your color dictionary
    def get_model_label(row):
        return f"L{row['n_layer']}H{row['n_head']}E{row['n_embd']}"
    df['model_label'] = df.apply(get_model_label, axis=1)
    
    # Get unique model architectures
    model_archs = df['model_label'].unique()
    print("Model architectures found:", model_archs)
    
    # Create subplots - one row for CPU, one for MPS
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Colors for different model architectures - updated to match the labels you're creating
    model_colors = {
    'L2H2E128': '#9467bd',    # Purple
    'L4H4E256': '#ff7f0e',    # Orange  
    'L6H6E384': '#2ca02c',    # Green
}
    
    # Fallback colors in case we have unexpected models
    fallback_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Plot for CPU device
    cpu_data = df[df['device'] == 'cpu']
    for i, arch in enumerate(model_archs):
        arch_data = cpu_data[cpu_data['model_label'] == arch]
        if not arch_data.empty:
            # Filter out zero throughput values for plotting AND trend line
            valid_arch_data = arch_data[arch_data['steps_per_second'] > 0]
            
            if not valid_arch_data.empty:
                color = model_colors.get(arch, fallback_colors[i % len(fallback_colors)])
                ax1.scatter(valid_arch_data['total_power'], valid_arch_data['steps_per_second'], 
                           color=color, label=arch, s=100, alpha=0.7, edgecolors='black')
                
                # Add trend line only if we have valid data points
                if len(valid_arch_data) > 1:
                    try:
                        z = np.polyfit(valid_arch_data['total_power'], valid_arch_data['steps_per_second'], 1)
                        p = np.poly1d(z)
                        x_range = np.linspace(valid_arch_data['total_power'].min(), valid_arch_data['total_power'].max(), 100)
                        ax1.plot(x_range, p(x_range), color=color, alpha=0.5, linestyle='--')
                        
                        # Optional: Add R² value to show fit quality
                        correlation = np.corrcoef(valid_arch_data['total_power'], valid_arch_data['steps_per_second'])[0,1]
                        r_squared = correlation ** 2
                        # You can add text annotation if desired:
                        # ax1.text(x_range.mean(), p(x_range).mean(), f'R²={r_squared:.2f}', 
                        #          fontsize=8, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
                        
                    except (ValueError, np.linalg.LinAlgError):
                        # Skip trend line if fitting fails (e.g., all same x values)
                        print(f"Could not fit trend line for {arch} on CPU")
                        pass
    
    ax1.set_xlabel('Total Power Consumption (Watts)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Throughput (Steps/second)', fontsize=12, fontweight='bold')
    ax1.set_title('Performance vs Power Efficiency - CPU Device\nThroughput vs Total Power by Model Architecture', 
                 fontsize=14, fontweight='bold')
    ax1.legend(title='Model Architecture', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)
    
    # Plot for MPS device
    mps_data = df[df['device'] == 'mps']
    for i, arch in enumerate(model_archs):
        arch_data = mps_data[mps_data['model_label'] == arch]
        if not arch_data.empty:
            # Filter out zero throughput values for plotting AND trend line
            valid_arch_data = arch_data[arch_data['steps_per_second'] > 0]
            
            if not valid_arch_data.empty:
                color = model_colors.get(arch, fallback_colors[i % len(fallback_colors)])
                ax2.scatter(valid_arch_data['total_power'], valid_arch_data['steps_per_second'], 
                           color=color, label=arch, s=100, alpha=0.7, edgecolors='black')
                
                # Add trend line only if we have valid data points
                if len(valid_arch_data) > 1:
                    try:
                        z = np.polyfit(valid_arch_data['total_power'], valid_arch_data['steps_per_second'], 1)
                        p = np.poly1d(z)
                        x_range = np.linspace(valid_arch_data['total_power'].min(), valid_arch_data['total_power'].max(), 100)
                        ax2.plot(x_range, p(x_range), color=color, alpha=0.5, linestyle='--')
                    except (ValueError, np.linalg.LinAlgError):
                        # Skip trend line if fitting fails (e.g., all same x values)
                        print(f"Could not fit trend line for {arch} on MPS")
                        pass
    
    ax2.set_xlabel('Total Power Consumption (Watts)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Throughput (Steps/second)', fontsize=12, fontweight='bold')
    ax2.set_title('Performance vs Power Efficiency - MPS Device\nThroughput vs Total Power by Model Architecture', 
                 fontsize=14, fontweight='bold')
    ax2.legend(title='Model Architecture', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    return fig

def plot_throughput_vs_energy_by_model(df, save_path=None):
    """
    Plot 3B: Throughput vs Energy by Model Architecture
    Two subplots (CPU and MPS devices) showing relationship between throughput and energy consumption
    """
    # Calculate energy consumption (Joules)
    df['total_energy'] = (df['avg_cpu_power'] + df['avg_gpu_power']) * df['training_time']
    
    # Create model architecture labels
    def get_model_label(row):
        return f"L{row['n_layer']}H{row['n_head']}E{row['n_embd']}"
    
    df['model_label'] = df.apply(get_model_label, axis=1)
    
    # Get unique model architectures
    model_archs = df['model_label'].unique()
    
    # Create subplots - one row for CPU, one for MPS
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Manual color assignment for consistent colors across plots

    model_colors = {
        'L2H2E128 (~0.4M)': '#9467bd',    # Blue
        'L4H4E256 (~3.2M)': '#ff7f0e',    # Orange
        'L6H6E384 (~10.0M)': '#2ca02c',   # Green
    }
    
    # Fallback colors if we have more models than defined
    fallback_colors = ['#9467bd', '#ff7f0e', '#2ca02c',]
    
    # Plot for CPU device
    cpu_data = df[df['device'] == 'cpu']
    for i, arch in enumerate(model_archs):
        arch_data = cpu_data[cpu_data['model_label'] == arch]
        if not arch_data.empty:
            # Filter out zero throughput values
            valid_arch_data = arch_data[arch_data['steps_per_second'] > 0]
            
            if not valid_arch_data.empty:
                # Get color - use predefined if available, otherwise fallback
                color = model_colors.get(arch, fallback_colors[i % len(fallback_colors)])
                
                ax1.scatter(valid_arch_data['total_energy'], valid_arch_data['steps_per_second'], 
                           color=color, label=arch, s=100, alpha=0.7, edgecolors='black')
                
                # Add trend line (only with valid data)
                if len(valid_arch_data) > 1:
                    try:
                        z = np.polyfit(valid_arch_data['total_energy'], valid_arch_data['steps_per_second'], 1)
                        p = np.poly1d(z)
                        x_range = np.linspace(valid_arch_data['total_energy'].min(), valid_arch_data['total_energy'].max(), 100)
                        ax1.plot(x_range, p(x_range), color=color, alpha=0.5, linestyle='--')
                    except (ValueError, np.linalg.LinAlgError):
                        # Skip trend line if fitting fails
                        pass
    
    ax1.set_xlabel('Total Energy Consumption (Joules)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Throughput (Steps/second)', fontsize=12, fontweight='bold')
    ax1.set_title('Throughput vs Energy Efficiency - CPU Device\nPerformance vs Total Energy by Model Architecture', 
                 fontsize=14, fontweight='bold')
    ax1.legend(title='Model Architecture', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)
    
    # Plot for MPS device
    mps_data = df[df['device'] == 'mps']
    for i, arch in enumerate(model_archs):
        arch_data = mps_data[mps_data['model_label'] == arch]
        if not arch_data.empty:
            # Filter out zero throughput values
            valid_arch_data = arch_data[arch_data['steps_per_second'] > 0]
            
            if not valid_arch_data.empty:
                # Get color - use predefined if available, otherwise fallback
                color = model_colors.get(arch, fallback_colors[i % len(fallback_colors)])
                
                ax2.scatter(valid_arch_data['total_energy'], valid_arch_data['steps_per_second'], 
                           color=color, label=arch, s=100, alpha=0.7, edgecolors='black')
                
                # Add trend line (only with valid data)
                if len(valid_arch_data) > 1:
                    try:
                        z = np.polyfit(valid_arch_data['total_energy'], valid_arch_data['steps_per_second'], 1)
                        p = np.poly1d(z)
                        x_range = np.linspace(valid_arch_data['total_energy'].min(), valid_arch_data['total_energy'].max(), 100)
                        ax2.plot(x_range, p(x_range), color=color, alpha=0.5, linestyle='--')
                    except (ValueError, np.linalg.LinAlgError):
                        # Skip trend line if fitting fails
                        pass
    
    ax2.set_xlabel('Total Energy Consumption (Joules)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Throughput (Steps/second)', fontsize=12, fontweight='bold')
    ax2.set_title('Throughput vs Energy Efficiency - MPS Device\nPerformance vs Total Energy by Model Architecture', 
                 fontsize=14, fontweight='bold')
    ax2.legend(title='Model Architecture', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    return fig

fig_3a = plot_throughput_vs_power_by_model(df, save_path='results/plot_3a_throughput_vs_power.png')
fig_3b = plot_throughput_vs_energy_by_model(df, save_path='results/plot_3b_throughput_vs_energy.png')
