import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, root_dir)
import utils.plot_helper as plot_helper
load_latest_results = plot_helper.load_latest_results
df, latest_file = load_latest_results()

def plot_batch_size_memory_vs_compute_bound_analysis(df, save_path=None):
    """
    Plot 6A: Memory-bound vs Compute-bound Analysis for Batch Size
    Analyzes how batch size affects performance differently on CPU vs MPS
    """
    
    # Focus on one representative model size across different batch sizes
    plot_data = df[
        (df['n_layer'] == 4) & 
        (df['n_head'] == 4) & 
        (df['n_embd'] == 256) &
        (df['block_size'] == 64)
    ].copy()
    
    # Calculate performance metrics
    plot_data['total_power'] = plot_data['avg_cpu_power'] + plot_data['avg_gpu_power']
    plot_data['energy_per_step'] = plot_data['total_power'] / plot_data['steps_per_second']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    devices = ['cpu', 'mps']
    colors = ['#d62728', '#2ca02c']
    
    # Get sorted batch sizes and prepare data
    batch_sizes = sorted(plot_data['batch_size'].unique())
    
    # Helper function to find intersections between two lines
    def find_line_intersections(x1, y1, x2, y2):
        """Find intersections between two lines defined by (x1,y1) and (x2,y2) points"""
        intersections = []
        
        # Create interpolation functions for both lines
        from scipy.interpolate import interp1d
        try:
            f1 = interp1d(x1, y1, kind='linear', bounds_error=False, fill_value='extrapolate')
            f2 = interp1d(x2, y2, kind='linear', bounds_error=False, fill_value='extrapolate')
            
            # Create dense x range for intersection detection
            x_min = max(min(x1), min(x2))
            x_max = min(max(x1), max(x2))
            if x_min < x_max:
                x_dense = np.linspace(x_min, x_max, 1000)
                y1_dense = f1(x_dense)
                y2_dense = f2(x_dense)
                
                # Find where the lines cross
                for i in range(len(x_dense) - 1):
                    if (y1_dense[i] - y2_dense[i]) * (y1_dense[i + 1] - y2_dense[i + 1]) <= 0:
                        # Linear interpolation to find exact intersection
                        x_int = x_dense[i] - (y1_dense[i] - y2_dense[i]) * (x_dense[i + 1] - x_dense[i]) / ((y1_dense[i + 1] - y2_dense[i + 1]) - (y1_dense[i] - y2_dense[i]))
                        y_int = f1(x_int)
                        intersections.append((x_int, y_int))
        except:
            # Fallback: simple approach if interpolation fails
            pass
            
        return intersections
    
    # Prepare data for intersection finding
    cpu_data = plot_data[plot_data['device'] == 'cpu'].sort_values('batch_size')
    mps_data = plot_data[plot_data['device'] == 'mps'].sort_values('batch_size')
    
    # Plot 1: Throughput vs Batch Size with break-even analysis
    throughput_intersections = []
    if not cpu_data.empty and not mps_data.empty:
        cpu_x = cpu_data['batch_size'].values
        cpu_y = cpu_data['steps_per_second'].values
        mps_x = mps_data['batch_size'].values
        mps_y = mps_data['steps_per_second'].values
        
        throughput_intersections = find_line_intersections(cpu_x, cpu_y, mps_x, mps_y)
    
    for i, device in enumerate(devices):
        device_data = plot_data[plot_data['device'] == device].sort_values('batch_size')
        if not device_data.empty:
            ax1.plot(device_data['batch_size'], device_data['steps_per_second'], 
                    marker='o', linewidth=2, color=colors[i], label=f'{device.upper()}')
            ax1.scatter(device_data['batch_size'], device_data['steps_per_second'],
                       color=colors[i], s=80, alpha=0.8)
    
    # Mark throughput intersections
    for x_int, y_int in throughput_intersections:
        ax1.axvline(x=x_int, color='blue', linestyle='--', alpha=0.7, linewidth=1)
        ax1.annotate(f'Break-even\n{x_int:.1f}', 
                    xy=(x_int, y_int), 
                    xytext=(10, 30), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
                    ha='center', va='bottom', fontweight='bold', color='blue',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    ax1.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Throughput (steps/second)', fontsize=11, fontweight='bold')
    ax1.set_title('Throughput vs Batch Size\n(Indicates Memory vs Compute Bound)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training Time vs Batch Size with break-even analysis
    time_intersections = []
    if not cpu_data.empty and not mps_data.empty:
        cpu_x = cpu_data['batch_size'].values
        cpu_y = cpu_data['training_time'].values
        mps_x = mps_data['batch_size'].values
        mps_y = mps_data['training_time'].values
        
        time_intersections = find_line_intersections(cpu_x, cpu_y, mps_x, mps_y)
    
    for i, device in enumerate(devices):
        device_data = plot_data[plot_data['device'] == device].sort_values('batch_size')
        if not device_data.empty:
            ax2.plot(device_data['batch_size'], device_data['training_time'], 
                    marker='s', linewidth=2, color=colors[i], label=f'{device.upper()}')
            ax2.scatter(device_data['batch_size'], device_data['training_time'],
                       color=colors[i], s=80, alpha=0.8)
    
    # Mark training time intersections
    for x_int, y_int in time_intersections:
        ax2.axvline(x=x_int, color='blue', linestyle='--', alpha=0.7, linewidth=1)
        ax2.annotate(f'Break-even\n{x_int:.1f}', 
                    xy=(x_int, y_int), 
                    xytext=(10, 30), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
                    ha='center', va='bottom', fontweight='bold', color='blue',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    ax2.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Training Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_title('Training Time vs Batch Size', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Energy per Step vs Batch Size
    energy_intersections = []
    if not cpu_data.empty and not mps_data.empty:
        cpu_x = cpu_data['batch_size'].values
        cpu_y = cpu_data['energy_per_step'].values
        mps_x = mps_data['batch_size'].values
        mps_y = mps_data['energy_per_step'].values
        
        energy_intersections = find_line_intersections(cpu_x, cpu_y, mps_x, mps_y)
    
    for i, device in enumerate(devices):
        device_data = plot_data[plot_data['device'] == device].sort_values('batch_size')
        if not device_data.empty:
            # Filter out infinite energy values (when throughput = 0)
            valid_data = device_data[device_data['steps_per_second'] > 0]
            if not valid_data.empty:
                ax3.plot(valid_data['batch_size'], valid_data['energy_per_step'], 
                        marker='^', linewidth=2, color=colors[i], label=f'{device.upper()}')
                ax3.scatter(valid_data['batch_size'], valid_data['energy_per_step'],
                           color=colors[i], s=80, alpha=0.8)
        
    # Mark training time intersections
    for x_int, y_int in energy_intersections:
        ax3.axvline(x=x_int, color='blue', linestyle='--', alpha=0.7, linewidth=1)
        ax3.annotate(f'Break-even\n{x_int:.1f}', 
                    xy=(x_int, y_int), 
                    xytext=(10, 30), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
                    ha='center', va='bottom', fontweight='bold', color='blue',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))

    ax3.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Energy per Step (Joules/step)', fontsize=11, fontweight='bold')
    ax3.set_title('Energy Efficiency vs Batch Size', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: MPS Speedup Ratio vs Batch Size - FIXED: Stop when CPU throughput becomes 0
    speedup_ratios = []
    valid_batch_sizes = []
    
    # Calculate speedup ratios only for valid batch sizes
    for batch_size in batch_sizes:
        cpu_data = plot_data[(plot_data['device'] == 'cpu') & (plot_data['batch_size'] == batch_size)]
        mps_data = plot_data[(plot_data['device'] == 'mps') & (plot_data['batch_size'] == batch_size)]
        
        if not cpu_data.empty and not mps_data.empty:
            cpu_throughput = cpu_data['steps_per_second'].iloc[0]
            mps_throughput = mps_data['steps_per_second'].iloc[0]
            
            # Stop when CPU throughput becomes 0 (OOM/computation failure)
            if cpu_throughput <= 0:
                break
                
            if cpu_throughput > 0:
                speedup_ratio = mps_throughput / cpu_throughput
                speedup_ratios.append(speedup_ratio)
                valid_batch_sizes.append(batch_size)
    
    # Only plot if we have valid data
    if valid_batch_sizes and speedup_ratios:
        # Plot as line with markers (consistent with other subplots)
        ax4.plot(valid_batch_sizes, speedup_ratios, marker='D', linewidth=2, 
                 color='purple', label='MPS/CPU Speedup Ratio', markersize=8)
        ax4.scatter(valid_batch_sizes, speedup_ratios, color='purple', s=80, alpha=0.8)
        
        # Find and mark break-even points (where speedup ratio = 1) - only in valid range
        speedup_break_even_points = []
        for i in range(len(valid_batch_sizes) - 1):
            if (speedup_ratios[i] - 1) * (speedup_ratios[i + 1] - 1) <= 0:
                # Linear interpolation to find exact break-even batch size
                x1, x2 = valid_batch_sizes[i], valid_batch_sizes[i + 1]
                y1, y2 = speedup_ratios[i] - 1, speedup_ratios[i + 1] - 1
                
                if y1 != y2:  # Avoid division by zero
                    break_even_x = x1 - y1 * (x2 - x1) / (y2 - y1)
                    speedup_break_even_points.append(break_even_x)
        
        # Mark break-even points with vertical lines and annotations
        for break_even_x in speedup_break_even_points:
            ax4.axvline(x=break_even_x, color='blue', linestyle='--', alpha=0.7, linewidth=1)
            ax4.annotate(f'Break-even\n{break_even_x:.1f}', 
                        xy=(break_even_x, 1), 
                        xytext=(10, 30), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
                        ha='center', va='bottom', fontweight='bold', color='blue',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        # Add horizontal break-even line
        ax4.axhline(y=1, color='blue', linestyle='--', alpha=0.7, linewidth=1, label='Break-even (1x)')
        
        # Add value labels on data points
        for i, (x, y) in enumerate(zip(valid_batch_sizes, speedup_ratios)):
            ax4.annotate(f'{y:.2f}x', 
                        xy=(x, y), 
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Set y-limits to show the data clearly
        y_min = min(min(speedup_ratios), 0.8)
        y_max = max(max(speedup_ratios), 1.2)
        ax4.set_ylim(y_min, y_max)
    else:
        ax4.text(0.5, 0.5, 'No valid speedup data\n(CPU throughput became 0)', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12, fontweight='bold')
    
    ax4.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax4.set_ylabel('MPS/CPU Speedup Ratio', fontsize=11, fontweight='bold')
    ax4.set_title('MPS Speedup vs Batch Size\n(Higher = More GPU Advantage)', fontsize=12, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    
    # Add analysis text box
    analysis_text = f"Break-even Analysis:\n"
    if speedup_break_even_points:
        for i, point in enumerate(speedup_break_even_points):
            analysis_text += f"Speedup BE: {point:.1f}\n"
    else:
        analysis_text += "No speedup break-even\n"
    
    if throughput_intersections:
        for x_int, y_int in throughput_intersections:
            analysis_text += f"Throughput BE: {x_int:.1f}\n"
    else:
        analysis_text += "No throughput break-even\n"
        
    if time_intersections:
        for x_int, y_int in time_intersections:
            analysis_text += f"Time BE: {x_int:.1f}"
    else:
        analysis_text += "No time break-even"
    
    ax4.text(0.02, 0.98, analysis_text, transform=ax4.transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    return fig

fig_6a = plot_batch_size_memory_vs_compute_bound_analysis(df, save_path='results/plot_6a_batch_size_memory_vs_compute_bound.png')
