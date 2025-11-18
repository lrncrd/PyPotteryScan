"""
Example: Using evaluate_parser as an imported function
Shows how to run parser evaluation programmatically with random sampling
"""

from evaluate_parser import run_evaluation

# Example 1: Run with default settings (10 random samplings per config)
print("Example 1: Default evaluation with random sampling")
print("-" * 60)

results = run_evaluation(
    ground_truth_path='GROUND_TRUTH.xlsx',
    output_dir='evaluation_results',
    num_iterations=10  # 10 random samplings per few-shot configuration
)

if results['success']:
    print(f"\n✅ Evaluation completed successfully!")
    print(f"   Output directory: {results['output_dir']}")
    print(f"   Configurations tested: {results['configurations_tested']}")
    print(f"   Random samplings per config: {results['num_iterations']}")
    print(f"   Total evaluations: {results['configurations_tested'] * results['num_iterations']}")
    
    # Access aggregated statistics
    for config_result in results['results']:
        num_ex = config_result['num_examples']
        iterations = config_result['iterations']
        
        avg_accs = [it['average_accuracy'] for it in iterations]
        print(f"\n   {num_ex} examples:")
        print(f"      Avg Accuracy: {sum(avg_accs)/len(avg_accs):.2f}% (±{(max(avg_accs)-min(avg_accs))/2:.2f})")
else:
    print(f"\n❌ Evaluation failed: {results.get('error', 'Unknown error')}")


# Example 2: Quick test with fewer iterations
print("\n\n" + "="*60)
print("Example 2: Quick test (3 iterations, 2 configs)")
print("-" * 60)

results = run_evaluation(
    ground_truth_path='GROUND_TRUTH.xlsx',
    output_dir='evaluation_quick',
    fewshot_counts=[3, 7],  # Only 2 configurations
    num_iterations=3  # Only 3 random samplings (faster)
)

if results['success']:
    print(f"\n✅ Quick evaluation completed!")
    print(f"   Total evaluations: {len(results['results']) * results['num_iterations']}")


# Example 3: Access detailed iteration data
print("\n\n" + "="*60)
print("Example 3: Analyze variance across iterations")
print("-" * 60)

results = run_evaluation(
    ground_truth_path='GROUND_TRUTH.xlsx',
    output_dir='evaluation_analysis',
    fewshot_counts=[5],  # Single configuration
    num_iterations=5
)

if results['success']:
    config = results['results'][0]
    iterations = config['iterations']
    
    print(f"\n5 few-shot examples with 5 random samplings:")
    print(f"\nIteration results:")
    for i, it in enumerate(iterations, 1):
        print(f"  Iteration {i}: Avg Acc={it['average_accuracy']:.2f}%, EMA={it['exact_match_accuracy']:.2f}%")
    
    # Calculate statistics
    import numpy as np
    accs = [it['average_accuracy'] for it in iterations]
    print(f"\nStatistics:")
    print(f"  Mean: {np.mean(accs):.2f}%")
    print(f"  Std:  {np.std(accs):.2f}%")
    print(f"  Min:  {np.min(accs):.2f}%")
    print(f"  Max:  {np.max(accs):.2f}%")


print("\n\n" + "="*60)
print("All examples completed!")
print("="*60)
