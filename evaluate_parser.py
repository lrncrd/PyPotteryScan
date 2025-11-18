"""
Parser Performance Evaluation Script
Evaluates LLM-based parsing accuracy using few-shot learning with different numbers of examples
Tests parser with 1, 3, 5, 7, 10 few-shot examples and measures all metrics
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import sys
import os

# Add app directory to path to import model_manager
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

try:
    from model_manager import ModelManager
    from config import Config
except ImportError:
    # If running as script, try alternative import
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_manager", 
                                                   os.path.join(os.path.dirname(__file__), 'app', 'model_manager.py'))
    model_manager_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_manager_module)
    ModelManager = model_manager_module.ModelManager
    
    spec_config = importlib.util.spec_from_file_location("config",
                                                          os.path.join(os.path.dirname(__file__), 'app', 'config.py'))
    config_module = importlib.util.module_from_spec(spec_config)
    spec_config.loader.exec_module(config_module)
    Config = config_module.Config

class ParserEvaluator:
    """Evaluates parser performance with few-shot learning"""
    
    def __init__(self, ground_truth_path: str, model_manager: ModelManager):
        """
        Initialize evaluator with ground truth data
        
        Args:
            ground_truth_path: Path to Excel file with 'parsing' sheet
            model_manager: Initialized ModelManager for running inference
        """
        # Load ground truth
        full_df = pd.read_excel(ground_truth_path, sheet_name='parsing')
        
        # Sample 60 observations randomly for faster evaluation
        if len(full_df) > 60:
            print(f"📊 Sampling 60 observations from {len(full_df)} total records for faster evaluation...")
            self.gt_df = full_df.sample(n=60, random_state=42).reset_index(drop=True)
            print(f"   Selected indices: {list(full_df.index[self.gt_df.index][:5])}... (showing first 5)")
        else:
            self.gt_df = full_df
        
        print(f"✅ Loaded ground truth: {len(self.gt_df)} records")
        print(f"   Columns: {list(self.gt_df.columns)}")
        
        self.model_manager = model_manager
        
        # Define target fields (customize based on your schema)
        self.fields = ['Inventory', 'Site', 'Year', 'US', 'Area', 'Cut', 'Sector', 'Notes', 'Phase']
        
        # Verify required columns
        if 'ocr_corrected' not in self.gt_df.columns:
            raise ValueError("Excel must have 'ocr_corrected' column with OCR text to parse")
        
        missing_fields = [f for f in self.fields if f not in self.gt_df.columns]
        if missing_fields:
            print(f"⚠️  Warning: Missing fields in ground truth: {missing_fields}")
            self.fields = [f for f in self.fields if f in self.gt_df.columns]
        
        print(f"📊 Evaluating fields: {self.fields}")
        
    def prepare_fewshot_examples(self, indices: List[int]) -> List[Dict]:
        """
        Prepare few-shot examples from ground truth records
        
        Args:
            indices: List of record indices to use as examples
            
        Returns:
            List of few-shot example dictionaries
        """
        examples = []
        for idx in indices:
            if idx >= len(self.gt_df):
                continue
                
            example = {
                'ocr_text': str(self.gt_df.iloc[idx]['ocr_corrected']),
                'parsed_data': {}
            }
            
            for field in self.fields:
                val = self.gt_df.iloc[idx][field]
                if pd.notna(val):
                    example['parsed_data'][field] = str(val)
                else:
                    example['parsed_data'][field] = ""
            
            examples.append(example)
        
        return examples
    
    def run_parser(self, ocr_text: str, fewshot_examples: List[Dict]) -> Dict:
        """
        Run parser on OCR text with few-shot examples
        
        Args:
            ocr_text: OCR text to parse
            fewshot_examples: List of few-shot examples
            
        Returns:
            Dictionary with parsed fields
        """
        result = self.model_manager.parse_with_fewshot(ocr_text, fewshot_examples, self.fields)
        return result
        
    def _normalize_value(self, val) -> str:
        """
        Normalize a value for comparison
        Removes spaces and special characters, keeping only letters and numbers
        This aggressive normalization is used ONLY for evaluation metrics
        """
        import re
        
        if pd.isna(val) or val is None:
            return ""
        
        # Convert to string and lowercase
        normalized = str(val).lower()
        
        # Remove all non-alphanumeric characters (keep only letters and numbers)
        normalized = re.sub(r'[^a-z0-9]', '', normalized)
        
        return normalized
    
    def compute_field_accuracy(self, predictions: List[Dict], test_indices: List[int], field: str) -> float:
        """
        Compute field-level accuracy for a specific field
        
        Args:
            predictions: List of prediction dictionaries
            test_indices: Indices of test records
            field: Field name
            
        Returns:
            Accuracy percentage (0-100)
        """
        if field not in self.fields:
            return 0.0
        
        n = len(test_indices)
        correct = 0
        
        for i, idx in enumerate(test_indices):
            if i >= len(predictions):
                break
                
            gt_val = self._normalize_value(self.gt_df.iloc[idx][field])
            pred_val = self._normalize_value(predictions[i].get(field, ""))
            
            if gt_val == pred_val:
                correct += 1
        
        accuracy = (correct / n) * 100 if n > 0 else 0.0
        return accuracy
    
    def compute_field_precision_recall_f1(self, predictions: List[Dict], test_indices: List[int], field: str) -> Tuple[float, float, float]:
        """
        Compute precision, recall, and F1 for a specific field
        Only considers non-empty values (ignores empty-empty matches)
        
        Args:
            predictions: List of prediction dictionaries
            test_indices: Indices of test records
            field: Field name
            
        Returns:
            Tuple of (precision, recall, F1) percentages
        """
        if field not in self.fields:
            return 0.0, 0.0, 0.0
        
        true_positives = 0  # GT has value, pred has value, they match
        false_positives = 0  # GT empty, pred has value OR GT has value, pred has wrong value
        false_negatives = 0  # GT has value, pred empty OR GT has value, pred has wrong value
        
        for i, idx in enumerate(test_indices):
            if i >= len(predictions):
                break
            
            gt_val = self._normalize_value(self.gt_df.iloc[idx][field])
            pred_val = self._normalize_value(predictions[i].get(field, ""))
            
            gt_has_value = bool(gt_val)
            pred_has_value = bool(pred_val)
            
            if gt_has_value and pred_has_value:
                if gt_val == pred_val:
                    true_positives += 1
                else:
                    # Wrong value: counts as both FP and FN
                    false_positives += 1
                    false_negatives += 1
            elif gt_has_value and not pred_has_value:
                # Missed extraction
                false_negatives += 1
            elif not gt_has_value and pred_has_value:
                # Hallucinated extraction
                false_positives += 1
            # else: both empty, don't count
        
        # Calculate metrics
        precision = (true_positives / (true_positives + false_positives) * 100) if (true_positives + false_positives) > 0 else 0.0
        recall = (true_positives / (true_positives + false_negatives) * 100) if (true_positives + false_negatives) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def compute_all_field_accuracies(self, predictions: List[Dict], test_indices: List[int]) -> Dict[str, float]:
        """
        Compute accuracy for all fields
        
        Args:
            predictions: List of prediction dictionaries
            test_indices: Indices of test records
            
        Returns:
            Dictionary mapping field names to accuracy percentages
        """
        accuracies = {}
        
        for field in self.fields:
            acc = self.compute_field_accuracy(predictions, test_indices, field)
            accuracies[field] = acc
        
        return accuracies
    
    def compute_all_field_metrics(self, predictions: List[Dict], test_indices: List[int]) -> Dict[str, Dict[str, float]]:
        """
        Compute precision, recall, F1 for all fields
        
        Args:
            predictions: List of prediction dictionaries
            test_indices: Indices of test records
            
        Returns:
            Dictionary mapping field names to metrics dict
        """
        metrics = {}
        
        for field in self.fields:
            precision, recall, f1 = self.compute_field_precision_recall_f1(predictions, test_indices, field)
            metrics[field] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return metrics
    
    def compute_partial_match_score(self, predictions: List[Dict], test_indices: List[int]) -> Tuple[float, float]:
        """
        Compute Partial Match Score: average percentage of correct fields per record
        Only considers fields that have values in ground truth
        
        Args:
            predictions: List of prediction dictionaries
            test_indices: Indices of test records
            
        Returns:
            Tuple of (average partial match %, std deviation)
        """
        partial_scores = []
        
        for i, idx in enumerate(test_indices):
            if i >= len(predictions):
                break
            
            # Count fields with values in GT
            gt_fields_with_values = 0
            correct_fields = 0
            
            for field in self.fields:
                gt_val = self._normalize_value(self.gt_df.iloc[idx][field])
                pred_val = self._normalize_value(predictions[i].get(field, ""))
                
                if gt_val:  # Only consider fields that should have values
                    gt_fields_with_values += 1
                    if gt_val == pred_val:
                        correct_fields += 1
            
            if gt_fields_with_values > 0:
                score = (correct_fields / gt_fields_with_values) * 100
                partial_scores.append(score)
        
        avg_score = np.mean(partial_scores) if partial_scores else 0.0
        std_score = np.std(partial_scores) if partial_scores else 0.0
        
        return avg_score, std_score
    
    def compute_exact_match_accuracy(self, predictions: List[Dict], test_indices: List[int]) -> Tuple[float, int]:
        """
        Compute Exact Match Accuracy (EMA) - all fields must be correct
        
        Args:
            predictions: List of prediction dictionaries
            test_indices: Indices of test records
            
        Returns:
            Tuple of (EMA percentage, number of exact matches)
        """
        n = len(test_indices)
        exact_matches = 0
        
        for i, idx in enumerate(test_indices):
            if i >= len(predictions):
                break
                
            all_correct = True
            
            for field in self.fields:
                gt_val = self._normalize_value(self.gt_df.iloc[idx][field])
                pred_val = self._normalize_value(predictions[i].get(field, ""))
                
                if gt_val != pred_val:
                    all_correct = False
                    break
            
            if all_correct:
                exact_matches += 1
        
        ema = (exact_matches / n) * 100 if n > 0 else 0.0
        return ema, exact_matches
    
    def compute_confusion_matrix(self, predictions: List[Dict], test_indices: List[int], field: str) -> Tuple[np.ndarray, List[str]]:
        """
        Compute confusion matrix for a specific field
        
        Args:
            predictions: List of prediction dictionaries
            test_indices: Indices of test records
            field: Field name
            
        Returns:
            Tuple of (confusion matrix, list of unique values)
        """
        if field not in self.fields:
            return np.array([]), []
        
        # Get unique values from both GT and predictions
        all_values = set()
        for i, idx in enumerate(test_indices):
            if i >= len(predictions):
                break
            gt_val = self._normalize_value(self.gt_df.iloc[idx][field])
            pred_val = self._normalize_value(predictions[i].get(field, ""))
            if gt_val:
                all_values.add(gt_val)
            if pred_val:
                all_values.add(pred_val)
        
        all_values = sorted(list(all_values))
        value_to_idx = {v: i for i, v in enumerate(all_values)}
        
        # Build confusion matrix
        k = len(all_values)
        confusion = np.zeros((k, k), dtype=int)
        
        for i, idx in enumerate(test_indices):
            if i >= len(predictions):
                break
            gt_val = self._normalize_value(self.gt_df.iloc[idx][field])
            pred_val = self._normalize_value(predictions[i].get(field, ""))
            
            if gt_val in value_to_idx and pred_val in value_to_idx:
                gt_idx = value_to_idx[gt_val]
                pred_idx = value_to_idx[pred_val]
                confusion[gt_idx, pred_idx] += 1
        
        return confusion, all_values
    
    def compute_field_confusion_matrix(self, predictions: List[Dict], test_indices: List[int]) -> Tuple[np.ndarray, List[str]]:
        """
        Compute field-level confusion matrix showing which fields are confused with each other
        Shows when information intended for field A is placed in field B
        
        Args:
            predictions: List of prediction dictionaries
            test_indices: Indices of test records
            
        Returns:
            Tuple of (confusion matrix [k x k], list of field names)
        """
        k = len(self.fields)
        confusion = np.zeros((k, k), dtype=int)
        field_to_idx = {f: i for i, f in enumerate(self.fields)}
        
        # For each test record
        for i, idx in enumerate(test_indices):
            if i >= len(predictions):
                break
            
            # Get all ground truth values (normalized)
            gt_values = {}
            for field in self.fields:
                val = self._normalize_value(self.gt_df.iloc[idx][field])
                if val:  # Only consider non-empty values
                    gt_values[field] = val
            
            # Get all predicted values (normalized)
            pred_values = {}
            for field in self.fields:
                val = self._normalize_value(predictions[i].get(field, ""))
                if val:  # Only consider non-empty values
                    pred_values[field] = val
            
            # For each ground truth value, find where it ended up in predictions
            for gt_field, gt_value in gt_values.items():
                # Check if this exact value appears in any prediction field
                found = False
                for pred_field, pred_value in pred_values.items():
                    if gt_value == pred_value:
                        # This ground truth value was placed in pred_field
                        gt_idx = field_to_idx[gt_field]
                        pred_idx = field_to_idx[pred_field]
                        confusion[gt_idx, pred_idx] += 1
                        found = True
                        break
                
                # If not found anywhere, it's a miss (could count as confusion with empty)
                if not found:
                    # Value was lost/not extracted
                    pass
        
        return confusion, self.fields
    
    def plot_confusion_matrix(self, predictions: List[Dict], test_indices: List[int], 
                             field: str, save_path: str = None):
        """
        Plot and optionally save confusion matrix for a field
        
        Args:
            predictions: List of prediction dictionaries
            test_indices: Indices of test records
            field: Field name
            save_path: Optional path to save the plot
        """
        confusion, labels = self.compute_confusion_matrix(predictions, test_indices, field)
        
        if confusion.size == 0:
            print(f"⚠️  No data for field: {field}")
            return
        
        plt.figure(figsize=(max(10, len(labels) * 0.8), max(8, len(labels) * 0.6)))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Count'})
        
        plt.title(f'Confusion Matrix for Field: {field}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Value', fontsize=12)
        plt.ylabel('Ground Truth Value', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved confusion matrix: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_field_confusion_matrix(self, predictions: List[Dict], test_indices: List[int], 
                                   save_path: str = None):
        """
        Plot field-level confusion matrix showing which fields are confused with each other
        
        Args:
            predictions: List of prediction dictionaries
            test_indices: Indices of test records
            save_path: Optional path to save the plot
        """
        confusion, labels = self.compute_field_confusion_matrix(predictions, test_indices)
        
        if confusion.size == 0:
            print(f"⚠️  No data for field confusion matrix")
            return
        
        plt.figure(figsize=(12, 10))
        
        # Normalize by row to show percentages
        row_sums = confusion.sum(axis=1, keepdims=True)
        confusion_pct = np.divide(confusion, row_sums, where=row_sums != 0) * 100
        
        # Create annotations showing both count and percentage
        annot = np.empty_like(confusion, dtype=object)
        for i in range(confusion.shape[0]):
            for j in range(confusion.shape[1]):
                count = int(confusion[i, j])
                pct = confusion_pct[i, j]
                if count > 0:
                    annot[i, j] = f'{count}\n({pct:.0f}%)'
                else:
                    annot[i, j] = ''
        
        sns.heatmap(confusion_pct, annot=annot, fmt='', cmap='YlOrRd', 
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Percentage of GT values'})
        
        plt.title('Field Confusion Matrix\n(Where does information from each field end up?)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Field (where the value was placed)', fontsize=12)
        plt.ylabel('Ground Truth Field (where the value should be)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved field confusion matrix: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_errors(self, predictions: List[Dict], test_indices: List[int]) -> pd.DataFrame:
        """
        Analyze and return detailed error information
        
        Args:
            predictions: List of prediction dictionaries
            test_indices: Indices of test records
            
        Returns:
            DataFrame with error details
        """
        errors = []
        
        for i, idx in enumerate(test_indices):
            if i >= len(predictions):
                break
                
            for field in self.fields:
                gt_val = self._normalize_value(self.gt_df.iloc[idx][field])
                pred_val = self._normalize_value(predictions[i].get(field, ""))
                
                if gt_val != pred_val:
                    errors.append({
                        'record_index': idx,
                        'field': field,
                        'ground_truth': self.gt_df.iloc[idx][field],
                        'predicted': predictions[i].get(field, ""),
                        'ocr_text': self.gt_df.iloc[idx]['ocr_corrected']
                    })
        
        return pd.DataFrame(errors)
    
    def evaluate_with_fewshot(self, num_examples: int, output_dir: Path, iteration: int = 1, seed: int = None):
        """
        Evaluate parser with specific number of few-shot examples
        
        Args:
            num_examples: Number of few-shot examples to use
            output_dir: Directory to save results
            iteration: Iteration number for this sampling
            seed: Random seed for reproducibility (if None, uses iteration as seed)
        """
        if seed is None:
            seed = iteration
        
        print(f"\n{'='*80}")
        print(f"EVALUATING WITH {num_examples} FEW-SHOT EXAMPLES - Iteration {iteration}")
        print(f"{'='*80}")
        
        # Random sampling of few-shot examples
        np.random.seed(seed)
        all_indices = np.arange(len(self.gt_df))
        np.random.shuffle(all_indices)
        
        # Select first N as few-shot, rest as test
        fewshot_indices = all_indices[:num_examples].tolist()
        test_indices = all_indices[num_examples:].tolist()
        
        print(f"  Few-shot examples (random): {len(fewshot_indices)}")
        print(f"  Test records: {len(test_indices)}")
        print(f"  Random seed: {seed}")
        
        # Prepare few-shot examples
        fewshot_examples = self.prepare_fewshot_examples(fewshot_indices)
        
        # Run parser on all test records
        predictions = []
        print(f"\n  Running parser on {len(test_indices)} test records...")
        
        for i, idx in enumerate(test_indices):
            if i % 10 == 0:
                print(f"    Progress: {i}/{len(test_indices)}")
            
            ocr_text = str(self.gt_df.iloc[idx]['ocr_corrected'])
            pred = self.run_parser(ocr_text, fewshot_examples)
            predictions.append(pred)
        
        print(f"    Completed: {len(predictions)}/{len(test_indices)}")
        
        # Compute all metrics
        print(f"\n  Computing metrics...")
        accuracies = self.compute_all_field_accuracies(predictions, test_indices)
        field_metrics = self.compute_all_field_metrics(predictions, test_indices)
        ema, exact_matches = self.compute_exact_match_accuracy(predictions, test_indices)
        partial_match, partial_std = self.compute_partial_match_score(predictions, test_indices)
        error_df = self.analyze_errors(predictions, test_indices)
        
        # Print results
        print(f"\n  {'='*60}")
        print(f"  FIELD-LEVEL METRICS")
        print(f"  {'='*60}")
        print(f"  {'Field':<15} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
        print(f"  {'-'*60}")
        for field in self.fields:
            acc = accuracies[field]
            prec = field_metrics[field]['precision']
            rec = field_metrics[field]['recall']
            f1 = field_metrics[field]['f1']
            print(f"  {field:<15} {acc:>6.2f}% {prec:>6.2f}% {rec:>6.2f}% {f1:>6.2f}%")
        
        avg_acc = np.mean(list(accuracies.values()))
        avg_prec = np.mean([m['precision'] for m in field_metrics.values()])
        avg_rec = np.mean([m['recall'] for m in field_metrics.values()])
        avg_f1 = np.mean([m['f1'] for m in field_metrics.values()])
        
        print(f"\n  {'Average':<15} {avg_acc:>6.2f}% {avg_prec:>6.2f}% {avg_rec:>6.2f}% {avg_f1:>6.2f}%")
        
        print(f"\n  {'='*60}")
        print(f"  OVERALL METRICS")
        print(f"  {'='*60}")
        print(f"    Exact Match Accuracy (EMA): {ema:.2f}%")
        print(f"      → All fields must be correct")
        print(f"    Partial Match Score (PMS): {partial_match:.2f}% ± {partial_std:.2f}%")
        print(f"      → Average % of correct non-empty fields per record")
        print(f"    Exact Matches: {exact_matches}/{len(test_indices)}")
        
        print(f"\n  {'='*60}")
        print(f"  ERROR SUMMARY")
        print(f"  {'='*60}")
        print(f"    Total errors: {len(error_df)}")
        
        if len(error_df) > 0:
            field_errors = error_df.groupby('field').size().sort_values(ascending=False)
            print(f"\n    Errors by field:")
            for field, count in field_errors.items():
                print(f"      {field:15s}: {count:4d} errors")
        
        # Save results
        results_dir = output_dir / f"fewshot_{num_examples}" / f"iteration_{iteration}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        metrics = {
            'num_fewshot_examples': num_examples,
            'iteration': iteration,
            'random_seed': seed,
            'fewshot_indices': fewshot_indices,
            'num_test_records': len(test_indices),
            'field_accuracies': accuracies,
            'field_precision': {f: field_metrics[f]['precision'] for f in self.fields},
            'field_recall': {f: field_metrics[f]['recall'] for f in self.fields},
            'field_f1': {f: field_metrics[f]['f1'] for f in self.fields},
            'average_accuracy': float(avg_acc),
            'average_precision': float(avg_prec),
            'average_recall': float(avg_rec),
            'average_f1': float(avg_f1),
            'exact_match_accuracy': float(ema),
            'exact_matches': int(exact_matches),
            'partial_match_score': float(partial_match),
            'partial_match_std': float(partial_std),
            'total_errors': len(error_df)
        }
        
        metrics_path = results_dir / "metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # Save predictions
        pred_df = pd.DataFrame(predictions)
        pred_df['gt_index'] = test_indices
        pred_path = results_dir / "predictions.csv"
        pred_df.to_csv(pred_path, index=False, encoding='utf-8')
        
        # Save error details
        if len(error_df) > 0:
            error_path = results_dir / "errors.csv"
            error_df.to_csv(error_path, index=False, encoding='utf-8')
        
        # Generate FIELD-LEVEL confusion matrix (only this one!)
        print(f"\n  Generating field-level confusion matrix...")
        field_conf_path = results_dir / "confusion_FIELDS.png"
        self.plot_field_confusion_matrix(predictions, test_indices, str(field_conf_path))
        
        print(f"\n✅ Results saved to: {results_dir}")
        
        return metrics


def run_evaluation(ground_truth_path: str, output_dir: str = 'evaluation_results', 
                   fewshot_counts: list = None, num_iterations: int = 10) -> dict:
    """
    Run parser evaluation with few-shot learning and multiple random samplings
    
    Args:
        ground_truth_path: Path to Excel file with ground truth (sheet: parsing)
        output_dir: Directory to save evaluation results
        fewshot_counts: List of few-shot example counts to test (default: [1, 3, 5, 7, 10])
        num_iterations: Number of random samplings per few-shot configuration (default: 10)
        
    Returns:
        Dictionary with summary results
    """
    if fewshot_counts is None:
        fewshot_counts = [1, 3, 5, 7, 10]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("PARSER PERFORMANCE EVALUATOR - FEW-SHOT LEARNING WITH RANDOM SAMPLING")
    print("="*80 + "\n")
    print(f"Configurations to test: {fewshot_counts}")
    print(f"Random samplings per configuration: {num_iterations}")
    print(f"Total evaluations: {len(fewshot_counts) * num_iterations}\n")
    
    # Initialize model manager with config
    print(f"📦 Loading configuration and initializing model manager...")
    config = Config.get_config_dict()
    model_manager = ModelManager(config)
    
    # Check/download models if needed
    if not model_manager.check_and_download_models():
        print("❌ Failed to download/check models")
        return {'success': False, 'error': 'Failed to download models'}
    
    print("✅ Models ready\n")
    
    # Initialize evaluator
    evaluator = ParserEvaluator(ground_truth_path, model_manager)
    
    # Store all results for comparison
    all_results = []
    
    # Evaluate with different numbers of few-shot examples
    for num_examples in fewshot_counts:
        if num_examples >= len(evaluator.gt_df):
            print(f"⚠️  Skipping {num_examples} examples (not enough data)")
            continue
        
        print(f"\n{'='*80}")
        print(f"TESTING WITH {num_examples} FEW-SHOT EXAMPLES")
        print(f"{'='*80}")
        
        # Run multiple iterations with random sampling
        iteration_results = []
        for iteration in range(1, num_iterations + 1):
            metrics = evaluator.evaluate_with_fewshot(num_examples, output_path, iteration)
            iteration_results.append(metrics)
        
        # Store all iteration results
        all_results.append({
            'num_examples': num_examples,
            'iterations': iteration_results
        })
    
    # Generate comparison report
    print(f"\n{'='*80}")
    print("FEW-SHOT LEARNING EFFICIENCY ANALYSIS")
    print(f"{'='*80}\n")
    
    # Aggregate results across iterations
    comparison = []
    for config_result in all_results:
        num_ex = config_result['num_examples']
        iterations = config_result['iterations']
        
        # Calculate statistics across iterations
        avg_accs = [it['average_accuracy'] for it in iterations]
        avg_precs = [it['average_precision'] for it in iterations]
        avg_recs = [it['average_recall'] for it in iterations]
        avg_f1s = [it['average_f1'] for it in iterations]
        emas = [it['exact_match_accuracy'] for it in iterations]
        pms = [it['partial_match_score'] for it in iterations]
        errors = [it['total_errors'] for it in iterations]
        
        comparison.append({
            'num_examples': num_ex,
            'avg_accuracy_mean': np.mean(avg_accs),
            'avg_accuracy_std': np.std(avg_accs),
            'avg_precision_mean': np.mean(avg_precs),
            'avg_precision_std': np.std(avg_precs),
            'avg_recall_mean': np.mean(avg_recs),
            'avg_recall_std': np.std(avg_recs),
            'avg_f1_mean': np.mean(avg_f1s),
            'avg_f1_std': np.std(avg_f1s),
            'ema_mean': np.mean(emas),
            'ema_std': np.std(emas),
            'pms_mean': np.mean(pms),
            'pms_std': np.std(pms),
            'errors_mean': np.mean(errors),
            'errors_std': np.std(errors),
            'num_iterations': len(iterations)
        })
    
    comp_df = pd.DataFrame(comparison)
    
    # Format for display
    display_df = comp_df.copy()
    display_df['accuracy'] = display_df.apply(lambda x: f"{x['avg_accuracy_mean']:.2f}±{x['avg_accuracy_std']:.2f}", axis=1)
    display_df['precision'] = display_df.apply(lambda x: f"{x['avg_precision_mean']:.2f}±{x['avg_precision_std']:.2f}", axis=1)
    display_df['recall'] = display_df.apply(lambda x: f"{x['avg_recall_mean']:.2f}±{x['avg_recall_std']:.2f}", axis=1)
    display_df['f1'] = display_df.apply(lambda x: f"{x['avg_f1_mean']:.2f}±{x['avg_f1_std']:.2f}", axis=1)
    display_df['ema'] = display_df.apply(lambda x: f"{x['ema_mean']:.2f}±{x['ema_std']:.2f}", axis=1)
    display_df['pms'] = display_df.apply(lambda x: f"{x['pms_mean']:.2f}±{x['pms_std']:.2f}", axis=1)
    display_df['errors'] = display_df.apply(lambda x: f"{x['errors_mean']:.1f}±{x['errors_std']:.1f}", axis=1)
    
    print("\n" + "="*100)
    print(f"{'Few-Shot':<10} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1':<15} {'EMA':<15} {'PMS':<15}")
    print("-"*100)
    for _, row in display_df.iterrows():
        print(f"{int(row['num_examples']):<10} {row['accuracy']:<15} {row['precision']:<15} {row['recall']:<15} {row['f1']:<15} {row['ema']:<15} {row['pms']:<15}")
    print("="*100)
    print("\nMetric Definitions:")
    print("  - Accuracy: Includes empty-empty matches (can be inflated)")
    print("  - Precision/Recall/F1: Only non-empty values (more stringent)")
    print("  - EMA: Exact Match Accuracy (all fields must be perfect)")
    print("  - PMS: Partial Match Score (avg % of correct non-empty fields per record)")
    print("="*100)
    
    # Save detailed comparison
    comp_path = output_path / "fewshot_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"\n✅ Comparison saved to: {comp_path}")
    
    # Save all raw iteration results
    all_iterations_data = []
    for config_result in all_results:
        for it_metrics in config_result['iterations']:
            all_iterations_data.append(it_metrics)
    
    all_iterations_df = pd.DataFrame(all_iterations_data)
    all_iterations_path = output_path / "all_iterations_metrics.csv"
    all_iterations_df.to_csv(all_iterations_path, index=False)
    print(f"✅ All iteration metrics saved to: {all_iterations_path}")
    
    # Plot learning curves with error bars - now with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Precision/Recall/F1
    axes[0].errorbar(comp_df['num_examples'], comp_df['avg_precision_mean'], 
                     yerr=comp_df['avg_precision_std'],
                     marker='o', linewidth=2, markersize=6, capsize=4, label='Precision')
    axes[0].errorbar(comp_df['num_examples'], comp_df['avg_recall_mean'], 
                     yerr=comp_df['avg_recall_std'],
                     marker='s', linewidth=2, markersize=6, capsize=4, label='Recall')
    axes[0].errorbar(comp_df['num_examples'], comp_df['avg_f1_mean'], 
                     yerr=comp_df['avg_f1_std'],
                     marker='^', linewidth=2, markersize=6, capsize=4, label='F1')
    axes[0].set_xlabel('Number of Few-Shot Examples', fontsize=11)
    axes[0].set_ylabel('Score (%)', fontsize=11)
    axes[0].set_title('Precision, Recall, F1 (non-empty values)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 2. EMA and PMS
    axes[1].errorbar(comp_df['num_examples'], comp_df['ema_mean'],
                     yerr=comp_df['ema_std'],
                     marker='o', linewidth=2, markersize=7, color='orange', 
                     capsize=4, label='EMA (all perfect)')
    axes[1].errorbar(comp_df['num_examples'], comp_df['pms_mean'],
                     yerr=comp_df['pms_std'],
                     marker='s', linewidth=2, markersize=7, color='green',
                     capsize=4, label='PMS (partial credit)')
    axes[1].set_xlabel('Number of Few-Shot Examples', fontsize=11)
    axes[1].set_ylabel('Score (%)', fontsize=11)
    axes[1].set_title('EMA (all perfect) vs PMS (partial credit)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # 3. Accuracy comparison (with and without empty matches)
    axes[2].errorbar(comp_df['num_examples'], comp_df['avg_accuracy_mean'], 
                     yerr=comp_df['avg_accuracy_std'],
                     marker='o', linewidth=2, markersize=7, capsize=4, 
                     label='Accuracy (with empty)', color='blue', alpha=0.6)
    axes[2].errorbar(comp_df['num_examples'], comp_df['avg_f1_mean'], 
                     yerr=comp_df['avg_f1_std'],
                     marker='s', linewidth=2, markersize=7, capsize=4, 
                     label='F1 (non-empty only)', color='red')
    axes[2].set_xlabel('Number of Few-Shot Examples', fontsize=11)
    axes[2].set_ylabel('Score (%)', fontsize=11)
    axes[2].set_title('Accuracy vs F1 Score', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
    axes[2].legend()
    
    plt.tight_layout()
    plot_path = output_path / "fewshot_learning_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Learning curves saved to: {plot_path}")
    plt.close()
    
    # Generate final summary report
    summary_path = output_path / "summary_report.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("PARSER EVALUATION SUMMARY - FEW-SHOT LEARNING WITH RANDOM SAMPLING\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Ground Truth File: {ground_truth_path}\n")
        f.write(f"Total Records: {len(evaluator.gt_df)}\n")
        f.write(f"Fields Evaluated: {', '.join(evaluator.fields)}\n")
        f.write(f"Random Samplings per Configuration: {num_iterations}\n\n")
        
        f.write("="*80 + "\n")
        f.write("FEW-SHOT LEARNING RESULTS (mean ± std)\n")
        f.write("="*80 + "\n\n")
        
        f.write(display_df[['num_examples', 'accuracy', 'precision', 'recall', 'f1', 'ema', 'pms', 'num_iterations']].to_string(index=False))
        f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("="*80 + "\n\n")
        
        # Find minimal examples for 90% F1
        above_90_f1 = comp_df[comp_df['avg_f1_mean'] >= 90]
        if len(above_90_f1) > 0:
            optimal_f1 = above_90_f1.iloc[0]['num_examples']
            f.write(f"✅ Minimal examples for >90% F1: {int(optimal_f1)}\n")
        else:
            f.write("⚠️  90% F1 threshold not reached with tested configurations\n")
        
        # Best configuration (based on F1)
        best_idx = comp_df['avg_f1_mean'].idxmax()
        best = comp_df.iloc[best_idx]
        f.write(f"🏆 Best configuration: {int(best['num_examples'])} examples\n")
        f.write(f"   - F1 Score: {best['avg_f1_mean']:.2f}% ± {best['avg_f1_std']:.2f}%\n")
        f.write(f"   - Precision: {best['avg_precision_mean']:.2f}% ± {best['avg_precision_std']:.2f}%\n")
        f.write(f"   - Recall: {best['avg_recall_mean']:.2f}% ± {best['avg_recall_std']:.2f}%\n")
        f.write(f"   - Exact Match Accuracy: {best['ema_mean']:.2f}% ± {best['ema_std']:.2f}%\n")
        f.write(f"   - Partial Match Score: {best['pms_mean']:.2f}% ± {best['pms_std']:.2f}%\n")
        f.write("\n")
        f.write("METRIC DEFINITIONS:\n")
        f.write("  - Accuracy: Includes empty-empty matches (can be inflated)\n")
        f.write("  - Precision/Recall/F1: Only non-empty values (more stringent)\n")
        f.write("  - EMA: Exact Match Accuracy (all fields must be perfect)\n")
        f.write("  - PMS: Partial Match Score (avg % of correct non-empty fields)\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✅ Summary report saved to: {summary_path}")
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}\n")
    
    # Return summary results
    return {
        'success': True,
        'output_dir': str(output_path),
        'total_records': len(evaluator.gt_df),
        'configurations_tested': len(all_results),
        'num_iterations': num_iterations,
        'results': all_results,
        'comparison': comparison,
        'files': {
            'summary_report': str(summary_path),
            'comparison_csv': str(comp_path),
            'all_iterations_csv': str(all_iterations_path),
            'learning_curves': str(plot_path)
        }
    }


def main():
    """Command-line interface for evaluation script"""
    parser = argparse.ArgumentParser(
        description='Evaluate parser performance with few-shot learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python evaluate_parser.py -g GROUND_TRUTH.xlsx -o evaluation_results
  
This will test the parser with 1, 3, 5, 7, and 10 few-shot examples.
The script will use the models configured in app/config.py

You can also import and use as a function:
  from evaluate_parser import run_evaluation
  results = run_evaluation('GROUND_TRUTH.xlsx', 'output_dir', [1, 3, 5])
        """
    )
    parser.add_argument('--ground-truth', '-g', required=True, 
                       help='Path to Excel file with ground truth (sheet: parsing)')
    parser.add_argument('--output-dir', '-o', default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--fewshot-counts', '-f', nargs='+', type=int,
                       default=[1, 3, 5, 7, 10],
                       help='Number of few-shot examples to test (default: 1 3 5 7 10)')
    parser.add_argument('--num-iterations', '-n', type=int, default=10,
                       help='Number of random samplings per configuration (default: 10)')
    
    args = parser.parse_args()
    
    # Call the main function
    result = run_evaluation(
        ground_truth_path=args.ground_truth,
        output_dir=args.output_dir,
        fewshot_counts=args.fewshot_counts,
        num_iterations=args.num_iterations
    )
    
    # Return exit code based on success
    return 0 if result.get('success', False) else 1


if __name__ == "__main__":
    exit(main())
