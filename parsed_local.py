# -*- coding: utf-8 -*-
"""
ocr_to_table_local_qwen3_interactive.py
Extracts structured data from OCR using a local Qwen3 model (via Hugging Face Transformers).
Allows the user to choose between 'guided' (few-shot) and 'free' parsing.
Saves the result in Excel (.xlsx).

Author: Lorenzo + Grok (Grok-4)
Language: British English
Sources: Hugging Face Transformers Docs
"""

import os
import time
import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------
# CONFIGURATION
# ------------------------
MODEL = "Qwen/Qwen3-1.7B"  # Local Qwen3 model from Hugging Face
CACHE_DIR = "./models"  # Directory to save the model locally
INPUT_CSV = "ceramic_data_2025-10-16.csv"
OUTPUT_XLSX = "parsed_output.xlsx"
PAUSE_SECONDS = 0.0  # No need for pause with local model, but can adjust if needed

# ------------------------
# PROMPT TEMPLATES
# ------------------------
SYSTEM_BASE = (
    "You are an assistant specialised in parsing short archaeological OCR lines. "
    "Return only one JSON object with these possible fields: "
    "Inventory, Site, Year, US, Area, Cut, Sector, Notes. "
    "If something is missing, set it to null. Avoid text outside JSON."
)

# Base few-shot examples (used only if user wants "guided" mode)
# Now read from file if exists
FEWSHOT_FILE = "fewshot_examples.json"
if os.path.exists(FEWSHOT_FILE):
    with open(FEWSHOT_FILE, 'r', encoding='utf-8') as f:
        FEWSHOT = json.load(f)
    print(f"✅ Few-shot loaded from {FEWSHOT_FILE}")
else:
    print(f"⚠️ File {FEWSHOT_FILE} not found, using default few-shot")
    FEWSHOT = [
        {"role": "user", "content": "INV. 256 MONTE S. GIULIA 2006 US 7 impasto chiaro con incisioni"},
        {"role": "assistant", "content": json.dumps({
            "Inventory": 256,
            "Site": "MONTE S. GIULIA", "Year": 2006, "US": 7,
            "Area": None, "Cut": None, "Sector": None, "Notes": "impasto chiaro con incisioni"
        })},
        {"role": "user", "content": "122 S. GIULIA 2015 US 81 964 ABSIDE SUD ceramica impressa"},
        {"role": "assistant", "content": json.dumps({
            "Inventory": 122,
            "Site": "S. GIULIA", "Year": 2015, "US": "81 964",
            "Area": None, "Cut": None, "Sector": "ABSIDE SUD", "Notes": "ceramica impressa"
        })}
    ]

# ------------------------
# COSTRUZIONE MESSAGGI
# ------------------------
def build_messages(line: str, guided: bool):
    """
    Build the chat messages dynamically based on mode:
    - guided=True  => include few-shot examples
    - guided=False => only generic system + user message
    """
    system = {"role": "system", "content": SYSTEM_BASE}
    user = {"role": "user", "content": f"Parse this OCR line and return JSON only: {line}"}

    if guided:
        messages = [system] + FEWSHOT + [user]
    else:
        messages = [system, user]
    return messages


# ------------------------
# MAIN
# ------------------------
def main():
    print("=== Archaeological OCR Parser via Local Qwen3 ===")
    mode = input("Do you want to use 'guided' parsing with examples (y/n)? ").strip().lower()
    guided = mode.startswith("y")

    # Load the model and tokenizer locally
    print(f"Loading model {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL, cache_dir=CACHE_DIR, torch_dtype="auto", device_map="auto")
    print("Model loaded successfully.")

    df = pd.read_csv(INPUT_CSV)

    if "ocr_corrected" not in df.columns:
        raise ValueError("The CSV must contain an 'ocr_corrected' column.")

    results = []
    for i, row in df.iterrows():
        text = str(row["ocr_corrected"])
        print(f"[{i+1}/{len(df)}] Parsing: {text}")

        messages = build_messages(text, guided)
        try:
            # Apply chat template
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # Non-thinking mode for simple parsing
            )
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

            # Generate response with temperature 0.0 (deterministic)
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False
            )
            content = tokenizer.decode(generated_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = {"raw_response": content}

            parsed["_ocr_original"] = text
            results.append(parsed)

        except Exception as e:
            print(f"  ⚠️ Error on row {i}: {e}")
            results.append({"_ocr_original": text, "error": str(e)})

        time.sleep(PAUSE_SECONDS)

    # Salvataggio in Excel
    out_df = pd.json_normalize(results)
    out_df.to_excel(OUTPUT_XLSX, index=False)
    print(f"\n✅ Excel file saved: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()