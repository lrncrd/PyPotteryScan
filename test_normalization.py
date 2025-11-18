"""
Test normalization function used in evaluation
Shows how different values are normalized for comparison
"""

import re
import pandas as pd

def normalize_value(val) -> str:
    """
    Normalize a value for comparison
    Removes spaces and special characters, keeping only letters and numbers
    """
    if pd.isna(val) or val is None:
        return ""
    
    # Convert to string and lowercase
    normalized = str(val).lower()
    
    # Remove all non-alphanumeric characters (keep only letters and numbers)
    normalized = re.sub(r'[^a-z0-9]', '', normalized)
    
    return normalized


# Test cases
test_cases = [
    ("US 1147", "us1147"),
    ("US1147", "us1147"),
    ("US-1147", "us1147"),
    ("US  1147", "us1147"),
    ("MONTALE 1998", "montale1998"),
    ("DIS.N.7", "disn7"),
    ("DIS. N. 7", "disn7"),
    ("Ib/1", "ib1"),
    ("Ib / 1", "ib1"),
    ("MON 98", "mon98"),
    ("", ""),
    (None, ""),
    ("123", "123"),
    ("ABC-123", "abc123"),
]

print("="*80)
print("NORMALIZATION TEST - Value Comparison")
print("="*80)
print()

print("Examples of how values are normalized for evaluation:\n")
print(f"{'Original Value':<25} → {'Normalized':<25} {'Expected':<25} {'Match'}")
print("-"*80)

all_pass = True
for original, expected in test_cases:
    normalized = normalize_value(original)
    match = "✅" if normalized == expected else "❌"
    if normalized != expected:
        all_pass = False
    
    original_str = repr(original) if original is not None else "None"
    print(f"{original_str:<25} → {normalized:<25} {expected:<25} {match}")

print("-"*80)

if all_pass:
    print("\n✅ All normalization tests passed!")
else:
    print("\n❌ Some tests failed!")

print("\n" + "="*80)
print("PRACTICAL EXAMPLES")
print("="*80)
print()

# Example comparisons that would MATCH
print("These pairs would be considered EQUAL in evaluation:")
matches = [
    ("US 1147", "US1147"),
    ("DIS. N. 7", "DIS.N.7"),
    ("Ib/1", "Ib 1"),
    ("MONTALE 1998", "montale1998"),
]

for val1, val2 in matches:
    norm1 = normalize_value(val1)
    norm2 = normalize_value(val2)
    print(f"  '{val1}' == '{val2}'  →  '{norm1}' == '{norm2}'  ✅")

print()

# Example comparisons that would NOT match
print("These pairs would be considered DIFFERENT in evaluation:")
mismatches = [
    ("US 1147", "US 1148"),
    ("DIS. N. 7", "DIS. N. 8"),
    ("Ib/1", "Ia/1"),
    ("MONTALE 1998", "MONTALE 2001"),
]

for val1, val2 in mismatches:
    norm1 = normalize_value(val1)
    norm2 = normalize_value(val2)
    print(f"  '{val1}' != '{val2}'  →  '{norm1}' != '{norm2}'  ❌")

print("\n" + "="*80)
print("\nThis normalization is used ONLY during evaluation metrics calculation.")
print("Original values are preserved in all output files (predictions.csv, errors.csv).")
print("="*80)
