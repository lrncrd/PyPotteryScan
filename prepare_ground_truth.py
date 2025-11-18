"""
Script to prepare ground truth Excel file from your existing data
Converts the CSV data into the format expected by evaluate_parser.py
"""

import pandas as pd
import argparse
from pathlib import Path

def prepare_ground_truth(input_csv: str, output_excel: str):
    """
    Prepare ground truth Excel file from CSV
    
    Args:
        input_csv: Path to input CSV file with ground truth columns
        output_excel: Path to output Excel file
    """
    print(f"\n{'='*80}")
    print("PREPARING GROUND TRUTH EXCEL FILE")
    print(f"{'='*80}\n")
    
    print(f"📂 Reading CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"✅ Loaded {len(df)} records")
    print(f"   Columns: {list(df.columns)}\n")
    
    # Define expected fields
    expected_fields = [
        'Inventory', 'Site', 'Year', 'US', 'Area', 
        'Cut', 'Sector', 'Notes', 'Phase', 'ocr_corrected'
    ]
    
    # Check for missing fields
    missing = [f for f in expected_fields if f not in df.columns]
    if missing:
        print(f"⚠️  Warning: Missing fields: {missing}")
        print(f"   These will be created with empty values\n")
        for field in missing:
            df[field] = ""
    
    # Reorder columns to have ocr_corrected last
    field_order = [f for f in expected_fields if f in df.columns]
    if 'ocr_corrected' in field_order:
        field_order.remove('ocr_corrected')
        field_order.append('ocr_corrected')
    
    df_ordered = df[field_order]
    
    print(f"📊 Creating Excel file...")
    print(f"   Output: {output_excel}")
    print(f"   Sheet: 'parsing'")
    print(f"   Fields: {field_order}\n")
    
    # Save to Excel with 'parsing' sheet
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df_ordered.to_excel(writer, sheet_name='parsing', index=False)
    
    print(f"✅ Ground truth Excel file created successfully!")
    print(f"   Records: {len(df_ordered)}")
    print(f"   Path: {output_excel}\n")
    
    # Show sample
    print("📋 Sample data (first 3 rows):")
    print(df_ordered.head(3).to_string())
    
    print(f"\n{'='*80}")
    print("You can now run the evaluation script:")
    print(f"python evaluate_parser.py -g {output_excel} -o evaluation_results")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare ground truth Excel file for parser evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python prepare_ground_truth.py -i ceramic_data_2025-10-16.csv -o GROUND_TRUTH.xlsx
        """
    )
    parser.add_argument('--input', '-i', required=True,
                       help='Path to input CSV file with ground truth data')
    parser.add_argument('--output', '-o', default='GROUND_TRUTH.xlsx',
                       help='Path to output Excel file (default: GROUND_TRUTH.xlsx)')
    
    args = parser.parse_args()
    
    # Check if input exists
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        return 1
    
    prepare_ground_truth(args.input, args.output)
    return 0


if __name__ == "__main__":
    exit(main())
