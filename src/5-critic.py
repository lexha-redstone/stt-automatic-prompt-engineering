import pandas as pd
import argparse
import jiwer
import os
import json
import re
from google import genai
from google.genai import types

def calculate_errors(df):
    """
    Calculates individual WER and CER for each row (sample) and adds them to the DataFrame.
    """
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
    ])
    
    wers = []
    cers = []
    
    for _, row in df.iterrows():
        r = str(row.get("transcription", ""))
        h = str(row.get("prediction", "")).replace("[EOT]", "").strip()
        
        if not r.strip():
            wers.append(0.0)
            cers.append(0.0)
            continue
            
        r_trans = transformation(r)
        h_trans = transformation(h)
        
        if not r_trans.strip():
            wers.append(0.0)
            cers.append(0.0)
            continue
        if not h_trans.strip():
            h_trans = " " 
            
        wers.append(jiwer.wer(r_trans, h_trans))
        cers.append(jiwer.cer(r_trans, h_trans))
        
    df['wer'] = wers
    df['cer'] = cers
    return df

def clean_json_response(text):
    text = text.strip()
    # Remove markdown blocks
    if text.startswith('```json'):
        text = text[7:]
    elif text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    return text.strip()

def main():
    parser = argparse.ArgumentParser(description="STT Error Critic for Generalization")
    parser.add_argument("--summary_csv", default="execution_log/evaluation_summary.csv", help="Path to evaluation summary CSV")
    parser.add_argument("--output_dir", default="execution_log/critic_results", help="Directory to save critic results")
    parser.add_argument("--project_id", type=str, default="my-argolis-prj", help="GCP Project ID")
    parser.add_argument("--location", type=str, default="global", help="GCP Location")
    parser.add_argument("--top_n", type=int, default=20, help="Number of worst examples to analyze")
    parser.add_argument("--model", type=str, default="gemini-3.1-pro-preview", help="Model to use for generating critics")
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    summary_csv_path = os.path.join(base_dir, args.summary_csv)
    output_dir_path = os.path.join(base_dir, args.output_dir)
    
    os.makedirs(output_dir_path, exist_ok=True)
    
    if not os.path.exists(summary_csv_path):
        print(f"Error: Summary CSV file not found: {summary_csv_path}")
        return

    df_summary = pd.read_csv(summary_csv_path)
    
    # 1. Filter only for cases where group is 'train'
    df_train = df_summary[df_summary['group'] == 'train']
    
    if df_train.empty:
        print("Warning: No data with group 'train' found in summary CSV.")
        return

    # Use only the latest record for each combination of lang, group, model, config, prompt_version (deduplication)
    subset_keys = ['lang', 'group', 'model', 'config']
    if 'prompt_version' in df_train.columns:
        subset_keys.append('prompt_version')
    df_train = df_train.drop_duplicates(subset=subset_keys, keep='last')
    
    client = genai.Client(vertexai=True, project=args.project_id, location=args.location)
    
    for _, row in df_train.iterrows():
        lang = row['lang']
        group = row['group']
        model_used = row['model']
        config_used = row['config']
        prompt_version = row.get('prompt_version', '')
        pv_clean = str(prompt_version).replace('.txt', '') if pd.notna(prompt_version) and prompt_version else 'default'
        local_csv = row['local_inference_csv_path']
        
        print(f"[{group}_{lang}] Model: {model_used}, Config: {config_used} analyzing critic...")
        
        if not os.path.isabs(local_csv):
            local_csv = os.path.join(base_dir, local_csv)
            
        if not os.path.exists(local_csv):
            print(f"  -> File not found, skipping: {local_csv}")
            continue
            
        print(f"  -> Loading evaluation data and calculating WER/CER...")
        df_local = pd.read_csv(local_csv)
        df_local = calculate_errors(df_local)
        
        # 2. Extract Top N samples with high WER and CER
        df_sorted = df_local.sort_values(by=['cer', 'wer'], ascending=[False, False])
        top_errors = df_sorted.head(args.top_n)
        
        error_cases_text = ""
        for i, (_, err_row) in enumerate(top_errors.iterrows(), 1):
            error_cases_text += f"## Case {i}\n"
            error_cases_text += f"Ground Truth: {str(err_row.get('transcription', ''))}\n"
            error_cases_text += f"Prediction:   {str(err_row.get('prediction', '')).replace('[EOT]', '').strip()}\n"
            error_cases_text += f"WER: {err_row.get('wer', 0.0):.2f}, CER: {err_row.get('cer', 0.0):.2f}\n"
            if "noisy_output_path" in err_row:
                error_cases_text += f"Audio File:   {os.path.basename(err_row['noisy_output_path'])}\n"
            error_cases_text += "\n"
            
        # 3. Constructing the prompt
        prompt = f"""You are an expert STT (Speech-to-Text) AI model critic.
I will provide you with the top {args.top_n} worst STT failure cases (highest WER/CER) for the language '{lang}'.
Your task is to analyze these errors and provide constructive, **generalizable** critiques to improve the STT prompt or configuration.

# Instructions:
1. Analyze each case to understand why the prediction failed compared to the ground truth.
2. Formulate an improvement strategy.
3. **CRITICAL:** Evaluate if the improvement strategy is generalizable to the broader dataset or if it is over-indexing on a highly specific, rare edge case. 
4. Filter out any highly specific, non-generalizable critiques. ONLY output the generalizable ones.
5. Group the generalizable critiques by error pattern. Do not just list cases one by one if they share the same root cause.

# Expected Output Format:
Output a pure JSON array containing the generalizable critiques. Do not wrap with markdown code blocks.
[
  {{
    "error_pattern": "Short description of the general error pattern",
    "affected_cases": "e.g., Case 1, Case 4, Case 7",
    "reason_for_failure": "Why the model generally fails here",
    "generalizable_improvement": "Actionable, generalized instructions for the system/prompt to fix this"
  }},
  ...
]

# Error Cases:
{error_cases_text}
"""
        
        try:
            print(f"  -> Requesting analysis from Gemini ({args.model})...")
            response = client.models.generate_content(
                model=args.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    # Force the model to return only JSON format
                    response_mime_type="application/json"
                )
            )
            
            clean_json = clean_json_response(response.text)
            critiques = json.loads(clean_json)
            
            if not critiques:
                print(f"  -> No generalizable critiques returned for {lang}.")
                continue
                
            # 5. Save to CSV
            df_critics = pd.DataFrame(critiques)
            df_critics.insert(0, 'lang', lang) # Add language column
            df_critics.insert(1, 'group', group)
            df_critics.insert(2, 'model', model_used)
            df_critics.insert(3, 'config', config_used)
            df_critics.insert(4, 'prompt_version', pv_clean)
            
            output_csv = os.path.join(output_dir_path, f"critic_{group}_{lang}_{model_used}_{config_used}_{pv_clean}.csv")
            df_critics.to_csv(output_csv, index=False, encoding='utf-8-sig')
            
            print(f"  -> ✅ Done: Generalizable critiques saved to ({output_csv})")
            
        except Exception as e:
            print(f"  -> ❌ Error occurred ({lang}): {e}")

if __name__ == "__main__":
    main()
