#!/usr/bin/env python3
"""
GridAdvisor Data Analysis Script
Analyzes profiling results and generates accuracy metrics
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def load_data(csv_file):
    """Load profiling results from CSV"""
    try:
        df = pd.read_csv(csv_file)
        print(f"✓ Loaded {len(df)} measurements")
        print(f"✓ Kernels: {df['kernel_name'].unique().tolist()}")
        return df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)

def find_actual_best(df):
    """Find actual best configuration for each kernel"""
    best_configs = {}
    
    for kernel in df['kernel_name'].unique():
        kernel_data = df[df['kernel_name'] == kernel]
        best_idx = kernel_data['time_ms'].idxmin()
        best_row = kernel_data.loc[best_idx]
        
        best_configs[kernel] = {
            'threads': int(best_row['threads_per_block']),
            'time_ms': float(best_row['time_ms']),
            'occupancy': float(best_row['occupancy'])
        }
    
    return best_configs

def calculate_accuracy(df, best_configs):
    """Calculate prediction accuracy"""
    results = []
    
    for kernel in df['kernel_name'].unique():
        kernel_data = df[df['kernel_name'] == kernel].iloc[0]  # Get first row for this kernel
        
        predicted = int(kernel_data['predicted_threads'])
        actual = best_configs[kernel]['threads']
        
        is_correct = (predicted == actual)
        error = abs(predicted - actual)
        
        results.append({
            'kernel': kernel,
            'type': kernel_data['kernel_type'],
            'predicted': predicted,
            'actual': actual,
            'correct': is_correct,
            'error': error,
            'best_time_ms': best_configs[kernel]['time_ms']
        })
    
    return pd.DataFrame(results)

def generate_summary(accuracy_df):
    """Generate summary statistics"""
    print("\n" + "="*60)
    print("ACCURACY SUMMARY")
    print("="*60 + "\n")
    
    total = len(accuracy_df)
    correct = accuracy_df['correct'].sum()
    accuracy = 100 * correct / total
    
    print(f"Overall Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print(f"Mean Prediction Error: ±{accuracy_df['error'].mean():.0f} threads")
    print(f"Median Prediction Error: ±{accuracy_df['error'].median():.0f} threads")
    
    print("\n" + "-"*60)
    print("Accuracy by Kernel Type:")
    print("-"*60)
    
    for ktype in accuracy_df['type'].unique():
        subset = accuracy_df[accuracy_df['type'] == ktype]
        type_correct = subset['correct'].sum()
        type_total = len(subset)
        type_acc = 100 * type_correct / type_total
        print(f"  {ktype:15s}: {type_correct}/{type_total} = {type_acc:.1f}%")
    
    print("\n" + "-"*60)
    print("Per-Kernel Results:")
    print("-"*60)
    print(f"{'Kernel':<20s} {'Type':<15s} {'Predicted':<10s} {'Actual':<10s} {'Correct':<8s}")
    print("-"*60)
    
    for _, row in accuracy_df.iterrows():
        status = "✓" if row['correct'] else "✗"
        print(f"{row['kernel']:<20s} {row['type']:<15s} {row['predicted']:<10d} "
              f"{row['actual']:<10d} {status:<8s}")
    
    print("="*60 + "\n")
    
    return {
        'overall_accuracy': accuracy,
        'mean_error': accuracy_df['error'].mean(),
        'median_error': accuracy_df['error'].median()
    }

def plot_results(df, best_configs, accuracy_df, output_dir):
    """Generate visualization plots"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    kernels = df['kernel_name'].unique()
    num_kernels = len(kernels)
    
    # 动态计算网格大小
    if num_kernels <= 6:
        rows, cols = 2, 3
    elif num_kernels <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 4, 3
    
    # Plot 1: Performance vs Thread Count for each kernel
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    fig.suptitle('Execution Time vs Thread Count', fontsize=16)
    
    # 如果只有一行或一列，axes不是2D数组
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, kernel in enumerate(kernels):
        if idx >= rows * cols:
            break
            
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        kernel_data = df[df['kernel_name'] == kernel]
        
        ax.plot(kernel_data['threads_per_block'], kernel_data['time_ms'], 
                marker='o', linewidth=2, markersize=8)
        
        # Mark best
        best_threads = best_configs[kernel]['threads']
        best_time = best_configs[kernel]['time_ms']
        ax.plot(best_threads, best_time, 'r*', markersize=20, label='Best')
        
        # Mark predicted
        predicted = accuracy_df[accuracy_df['kernel'] == kernel]['predicted'].values[0]
        pred_data = kernel_data[kernel_data['threads_per_block'] == predicted]
        if not pred_data.empty:
            ax.plot(predicted, pred_data['time_ms'].values[0], 
                   'gs', markersize=12, label='Predicted')
        
        ax.set_xlabel('Threads per Block')
        ax.set_ylabel('Time (ms)')
        ax.set_title(kernel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(num_kernels, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_curves.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'performance_curves.png'}")
    plt.close()
    
    # Plot 2: Prediction Accuracy Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    kernels = accuracy_df['kernel'].tolist()
    colors = ['green' if c else 'red' for c in accuracy_df['correct']]
    
    ax.bar(range(len(kernels)), [1]*len(kernels), color=colors, alpha=0.7)
    ax.set_xticks(range(len(kernels)))
    ax.set_xticklabels(kernels, rotation=45, ha='right')
    ax.set_ylabel('Prediction Accuracy')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Wrong', 'Correct'])
    ax.set_title('Analytical Model Prediction Accuracy per Kernel')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add accuracy percentage
    accuracy_pct = 100 * accuracy_df['correct'].sum() / len(accuracy_df)
    ax.text(0.5, 0.95, f'Overall Accuracy: {accuracy_pct:.1f}%',
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_by_kernel.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'accuracy_by_kernel.png'}")
    plt.close()
    
    # Plot 3: Occupancy vs Performance
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for kernel in df['kernel_name'].unique():
        kernel_data = df[df['kernel_name'] == kernel]
        ax.scatter(kernel_data['occupancy'], kernel_data['time_ms'], 
                  label=kernel, alpha=0.6, s=100)
    
    ax.set_xlabel('Occupancy', fontsize=12)
    ax.set_ylabel('Execution Time (ms)', fontsize=12)
    ax.set_title('Occupancy vs Execution Time', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'occupancy_vs_time.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'occupancy_vs_time.png'}")
    plt.close()

def save_report(summary, accuracy_df, output_dir):
    """Save text report"""
    output_dir = Path(output_dir)
    report_file = output_dir / 'analysis_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("GridAdvisor - Analysis Report\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Overall Accuracy: {summary['overall_accuracy']:.1f}%\n")
        f.write(f"Mean Error: ±{summary['mean_error']:.0f} threads\n")
        f.write(f"Median Error: ±{summary['median_error']:.0f} threads\n\n")
        
        f.write("-"*60 + "\n")
        f.write("Per-Kernel Results:\n")
        f.write("-"*60 + "\n")
        
        for _, row in accuracy_df.iterrows():
            status = "CORRECT" if row['correct'] else "WRONG"
            f.write(f"{row['kernel']:<20s} Pred:{row['predicted']:4d} "
                   f"Actual:{row['actual']:4d} [{status}]\n")
    
    print(f"✓ Saved: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze GridAdvisor profiling results')
    parser.add_argument('--input', '-i', 
                       default='data/profiling_results.csv',
                       help='Input CSV file (default: data/profiling_results.csv)')
    parser.add_argument('--output', '-o',
                       default='data/analysis',
                       help='Output directory (default: data/analysis)')
    args = parser.parse_args()
    
    csv_file = args.input
    output_dir = args.output
    
    print("\n" + "="*60)
    print("GridAdvisor - Data Analysis")
    print("="*60 + "\n")
    print(f"Input file: {csv_file}")
    print(f"Output dir: {output_dir}\n")

    print("\n" + "="*60)
    print("GridAdvisor - Data Analysis")
    print("="*60 + "\n")
    
    # Load data
    df = load_data(csv_file)
    
    # Find best configurations
    print("\nFinding best configurations...")
    best_configs = find_actual_best(df)
    
    # Calculate accuracy
    print("Calculating prediction accuracy...")
    accuracy_df = calculate_accuracy(df, best_configs)
    
    # Generate summary
    summary = generate_summary(accuracy_df)
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_results(df, best_configs, accuracy_df, output_dir)
    
    # Save report
    save_report(summary, accuracy_df, output_dir)
    
    print("\n" + "="*60)
    print("✓ Analysis completed!")
    print("="*60 + "\n")
    print(f"Results saved to: {output_dir}/")
    print("  - performance_curves.png")
    print("  - accuracy_by_kernel.png")
    print("  - occupancy_vs_time.png")
    print("  - analysis_report.txt")
    print()

if __name__ == '__main__':
    main()
    