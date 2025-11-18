#!/usr/bin/env python3
"""
Compare results across different problem sizes
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_all_experiments():
    """Load all experiment CSVs"""
    data = {}
    sizes = ['small', 'medium', 'large', 'xlarge']
    
    for size in sizes:
        try:
            df = pd.read_csv(f'data/profiling_{size}.csv')
            data[size] = df
            print(f"✓ Loaded {size}: {len(df)} measurements")
        except FileNotFoundError:
            print(f"✗ Missing: profiling_{size}.csv")
    
    return data

def analyze_scalability(data):
    """Analyze how performance scales with problem size"""
    print("\n" + "="*60)
    print("SCALABILITY ANALYSIS")
    print("="*60 + "\n")
    
    results = []
    
    for kernel in data['medium']['kernel_name'].unique():
        print(f"\n--- {kernel} ---")
        
        for size_name, df in data.items():
            kernel_data = df[df['kernel_name'] == kernel]
            if len(kernel_data) == 0:
                continue
                
            best_time = kernel_data['time_ms'].min()
            best_config = kernel_data.loc[kernel_data['time_ms'].idxmin()]
            
            results.append({
                'kernel': kernel,
                'size': size_name,
                'best_time_ms': best_time,
                'best_threads': int(best_config['threads_per_block']),
                'predicted_threads': int(best_config['predicted_threads'])
            })
            
            print(f"  {size_name:10s}: Best={best_time:.4f}ms @ {int(best_config['threads_per_block'])} threads")
    
    return pd.DataFrame(results)

def plot_scalability(scalability_df, output_dir):
    """Plot scalability curves"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    kernels = scalability_df['kernel'].unique()
    size_order = ['small', 'medium', 'large', 'xlarge']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Performance Scalability', fontsize=16)
    
    for idx, kernel in enumerate(kernels[:6]):
        ax = axes[idx // 3, idx % 3]
        kernel_data = scalability_df[scalability_df['kernel'] == kernel]
        
        # Sort by size
        kernel_data['size_order'] = kernel_data['size'].map(
            {s: i for i, s in enumerate(size_order)}
        )
        kernel_data = kernel_data.sort_values('size_order')
        
        # Plot time
        ax.plot(kernel_data['size'], kernel_data['best_time_ms'], 
                marker='o', linewidth=2, markersize=8, label='Best Time')
        
        ax.set_xlabel('Problem Size')
        ax.set_ylabel('Time (ms)')
        ax.set_title(kernel)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scalability_curves.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'scalability_curves.png'}")
    plt.close()

def analyze_prediction_accuracy(data):
    """Check if prediction accuracy changes with problem size"""
    print("\n" + "="*60)
    print("PREDICTION ACCURACY VS PROBLEM SIZE")
    print("="*60 + "\n")
    
    for size_name, df in data.items():
        correct = 0
        total = 0
        
        for kernel in df['kernel_name'].unique():
            kernel_data = df[df['kernel_name'] == kernel]
            
            # Find actual best
            best_idx = kernel_data['time_ms'].idxmin()
            actual_best = int(kernel_data.loc[best_idx, 'threads_per_block'])
            predicted = int(kernel_data.iloc[0]['predicted_threads'])
            
            if actual_best == predicted:
                correct += 1
            total += 1
        
        accuracy = 100 * correct / total
        print(f"{size_name:10s}: {correct}/{total} = {accuracy:.1f}%")

def plot_optimal_threads_vs_size(scalability_df, output_dir):
    """Plot how optimal thread count changes with problem size"""
    output_dir = Path(output_dir)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    size_order = ['small', 'medium', 'large', 'xlarge']
    size_map = {s: i for i, s in enumerate(size_order)}
    
    for kernel in scalability_df['kernel'].unique():
        kernel_data = scalability_df[scalability_df['kernel'] == kernel]
        kernel_data = kernel_data.sort_values(
            'size', key=lambda x: x.map(size_map)
        )
        
        ax.plot(kernel_data['size'], kernel_data['best_threads'], 
                marker='o', linewidth=2, markersize=8, label=kernel)
    
    ax.set_xlabel('Problem Size', fontsize=12)
    ax.set_ylabel('Optimal Threads per Block', fontsize=12)
    ax.set_title('Optimal Thread Count vs Problem Size', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'optimal_threads_vs_size.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'optimal_threads_vs_size.png'}")
    plt.close()

def main():
    print("\n" + "="*60)
    print("GridAdvisor - Multi-Experiment Analysis")
    print("="*60 + "\n")
    
    # Load data
    data = load_all_experiments()
    
    if len(data) == 0:
        print("✗ No experiment data found!")
        return
    
    # Analyze scalability
    scalability_df = analyze_scalability(data)
    
    # Analyze prediction accuracy
    analyze_prediction_accuracy(data)
    
    # Generate plots
    output_dir = 'data/comparison'
    plot_scalability(scalability_df, output_dir)
    plot_optimal_threads_vs_size(scalability_df, output_dir)
    
    # Save summary
    scalability_df.to_csv(f'{output_dir}/scalability_summary.csv', index=False)
    print(f"✓ Saved: {output_dir}/scalability_summary.csv")
    
    print("\n" + "="*60)
    print("✓ Comparison analysis completed!")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
    