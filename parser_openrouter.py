# -*- coding: utf-8 -*-
"""
ocr_to_table_openrouter_interactive.py
Extracts structured data from OCR using an OpenRouter model (via new OpenAI SDK).
Allows the user to choose between 'guided' (few-shot) and 'free' parsing.
Saves the result in Excel (.xlsx).

Author: Lorenzo + ChatGPT (GPT-5)
Language: British English
Sources: OpenRouter Docs (Quickstart / Python SDK)
"""

import os
import time
import json
import pandas as pd
from openai import OpenAI

# ------------------------
# CONFIGURATION
# ------------------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") or "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"
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "openai/gpt-oss-20b:free"  # modello free
INPUT_CSV = "ceramic_data_2025-10-16.csv"
OUTPUT_XLSX = "parsed_output.xlsx"
PAUSE_SECONDS = 0.8
HTTP_REFERER = "http://localhost"
X_TITLE = "PyOCRParser"

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
    print("=== Archaeological OCR Parser via OpenRouter ===")
    mode = input("Do you want to use 'guided' parsing with examples (y/n)? ").strip().lower()
    guided = mode.startswith("y")

    if OPENROUTER_API_KEY is None or "<INSERISCI" in OPENROUTER_API_KEY:
        raise ValueError("Please insert or set your OPENROUTER_API_KEY.")

    client = OpenAI(base_url=BASE_URL, api_key=OPENROUTER_API_KEY)
    df = pd.read_csv(INPUT_CSV)

    if "ocr_corrected" not in df.columns:
        raise ValueError("The CSV must contain an 'ocr_corrected' column.")

    results = []
    for i, row in df.iterrows():
        text = str(row["ocr_corrected"])
        print(f"[{i+1}/{len(df)}] Parsing: {text}")

        messages = build_messages(text, guided)
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.0,
                extra_headers={
                    "HTTP-Referer": HTTP_REFERER,
                    "X-Title": X_TITLE
                },
            )
            content = completion.choices[0].message.content
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
