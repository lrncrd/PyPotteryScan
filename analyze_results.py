"""
Additional analysis utilities for parser evaluation results
Use these functions to create custom visualizations and statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


def plot_field_accuracy_comparison(results_dir: str):
    """
    Plot field-by-field accuracy across different few-shot configurations
    
    Args:
        results_dir: Path to evaluation_results directory
    """
    results_path = Path(results_dir)
    
    # Load all metrics
    all_metrics = []
    for fewshot_dir in sorted(results_path.glob("fewshot_*")):
        metrics_file = fewshot_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                all_metrics.append(metrics)
    
    if not all_metrics:
        print("No metrics found!")
        return
    
    # Get all fields
    fields = list(all_metrics[0]['field_accuracies'].keys())
    
    # Create DataFrame
    data = []
    for metrics in all_metrics:
        row = {'num_examples': metrics['num_fewshot_examples']}
        row.update(metrics['field_accuracies'])
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = df['num_examples']
    for field in fields:
        ax.plot(x, df[field], marker='o', linewidth=2, markersize=6, label=field)
    
    ax.set_xlabel('Number of Few-Shot Examples', fontsize=13)
    ax.set_ylabel('Field Accuracy (%)', fontsize=13)
    ax.set_title('Field-Level Accuracy by Few-Shot Configuration', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90% threshold')
    
    plt.tight_layout()
    
    output_path = results_path / "field_accuracy_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def analyze_error_patterns(results_dir: str, fewshot_config: int = 5):
    """
    Analyze error patterns for a specific few-shot configuration
    
    Args:
        results_dir: Path to evaluation_results directory
        fewshot_config: Number of few-shot examples to analyze
    """
    errors_path = Path(results_dir) / f"fewshot_{fewshot_config}" / "errors.csv"
    
    if not errors_path.exists():
        print(f"Error file not found: {errors_path}")
        return
    
    errors_df = pd.read_csv(errors_path)
    
    print(f"\n{'='*80}")
    print(f"ERROR PATTERN ANALYSIS - {fewshot_config} Few-Shot Examples")
    print(f"{'='*80}\n")
    
    print(f"Total errors: {len(errors_df)}")
    print(f"Affected records: {errors_df['record_index'].nunique()}\n")
    
    # Errors by field
    print("Errors by field:")
    field_errors = errors_df.groupby('field').size().sort_values(ascending=False)
    for field, count in field_errors.items():
        pct = (count / len(errors_df)) * 100
        print(f"  {field:15s}: {count:4d} ({pct:5.1f}%)")
    
    # Most common errors per field
    print("\n" + "="*80)
    print("Most common error patterns:")
    print("="*80 + "\n")
    
    for field in field_errors.head(3).index:
        print(f"\n{field}:")
        field_errors_subset = errors_df[errors_df['field'] == field]
        error_patterns = field_errors_subset.groupby(['ground_truth', 'predicted']).size().sort_values(ascending=False).head(5)
        
        for (gt, pred), count in error_patterns.items():
            print(f"  '{gt}' → '{pred}': {count} times")
    
    print("\n" + "="*80)


def compare_configurations_table(results_dir: str):
    """
    Create a detailed comparison table for all configurations
    
    Args:
        results_dir: Path to evaluation_results directory
    """
    results_path = Path(results_dir)
    
    # Load all metrics
    all_data = []
    for fewshot_dir in sorted(results_path.glob("fewshot_*")):
        metrics_file = fewshot_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                
                row = {
                    'Examples': metrics['num_fewshot_examples'],
                    'Test Records': metrics['num_test_records'],
                    'Avg Accuracy': f"{metrics['average_accuracy']:.2f}%",
                    'EMA': f"{metrics['exact_match_accuracy']:.2f}%",
                    'Exact Matches': f"{metrics['exact_matches']}/{metrics['num_test_records']}",
                    'Total Errors': metrics['total_errors']
                }
                
                # Add top 3 field accuracies
                sorted_fields = sorted(metrics['field_accuracies'].items(), 
                                     key=lambda x: x[1], reverse=True)
                for i, (field, acc) in enumerate(sorted_fields[:3]):
                    row[f'Best Field {i+1}'] = f"{field} ({acc:.1f}%)"
                
                all_data.append(row)
    
    df = pd.DataFrame(all_data)
    
    print(f"\n{'='*80}")
    print("DETAILED CONFIGURATION COMPARISON")
    print(f"{'='*80}\n")
    print(df.to_string(index=False))
    print(f"\n{'='*80}\n")
    
    # Save to CSV
    output_path = results_path / "detailed_comparison.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path}")


def export_for_latex(results_dir: str):
    """
    Export results in LaTeX table format
    
    Args:
        results_dir: Path to evaluation_results directory
    """
    comp_path = Path(results_dir) / "fewshot_comparison.csv"
    
    if not comp_path.exists():
        print(f"Comparison file not found: {comp_path}")
        return
    
    df = pd.read_csv(comp_path)
    
    # Format for LaTeX
    latex_str = "\\begin{table}[ht]\n"
    latex_str += "\\centering\n"
    latex_str += "\\caption{Few-Shot Learning Performance Evaluation}\n"
    latex_str += "\\label{tab:fewshot_results}\n"
    latex_str += "\\begin{tabular}{c|ccc}\n"
    latex_str += "\\hline\n"
    latex_str += "Few-Shot & Average & Exact Match & Total \\\\\n"
    latex_str += "Examples & Accuracy (\\%) & Accuracy (\\%) & Errors \\\\\n"
    latex_str += "\\hline\n"
    
    for _, row in df.iterrows():
        latex_str += f"{int(row['num_examples'])} & "
        latex_str += f"{row['avg_accuracy']:.2f} & "
        latex_str += f"{row['ema']:.2f} & "
        latex_str += f"{int(row['total_errors'])} \\\\\n"
    
    latex_str += "\\hline\n"
    latex_str += "\\end{tabular}\n"
    latex_str += "\\end{table}\n"
    
    output_path = Path(results_dir) / "results_table.tex"
    with open(output_path, 'w') as f:
        f.write(latex_str)
    
    print(f"\n{'='*80}")
    print("LATEX TABLE")
    print(f"{'='*80}\n")
    print(latex_str)
    print(f"{'='*80}\n")
    print(f"✅ Saved: {output_path}")


def main():
    """Run all additional analyses"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Additional analysis utilities')
    parser.add_argument('--results-dir', '-r', default='evaluation_results',
                       help='Path to evaluation results directory')
    parser.add_argument('--plot-fields', action='store_true',
                       help='Plot field-by-field accuracy comparison')
    parser.add_argument('--analyze-errors', type=int, metavar='N',
                       help='Analyze error patterns for N few-shot examples')
    parser.add_argument('--comparison-table', action='store_true',
                       help='Generate detailed comparison table')
    parser.add_argument('--latex', action='store_true',
                       help='Export results in LaTeX format')
    parser.add_argument('--all', action='store_true',
                       help='Run all analyses')
    
    args = parser.parse_args()
    
    if args.all or args.plot_fields:
        plot_field_accuracy_comparison(args.results_dir)
    
    if args.all or args.analyze_errors is not None:
        config = args.analyze_errors if args.analyze_errors is not None else 5
        analyze_error_patterns(args.results_dir, config)
    
    if args.all or args.comparison_table:
        compare_configurations_table(args.results_dir)
    
    if args.all or args.latex:
        export_for_latex(args.results_dir)
    
    if not (args.plot_fields or args.analyze_errors is not None or 
            args.comparison_table or args.latex or args.all):
        parser.print_help()


if __name__ == "__main__":
    main()
