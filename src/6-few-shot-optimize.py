import pandas as pd
import argparse
import jiwer
import os
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

def extract_prompt_from_response(text):
    """
    Extracts the optimized prompt from the Gemini response.
    """
    # 1. ```prompt or ```text or ```markdown block
    blocks = re.findall(r'```(?:prompt|text|markdown)?\n(.*?)\n```', text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    
    # 2. '3. **Optimized Prompt Proposal**' 
    match = re.search(r'3\.\s*\*\*Optimized Prompt Proposal\*\*(.*)', text, re.DOTALL)
    if match:
        # Check for markdown block
        sub_blocks = re.findall(r'```(?:.*?)\n(.*?)\n```', match.group(1), re.DOTALL)
        if sub_blocks:
            return sub_blocks[-1].strip()
        return match.group(1).strip()
        
    return text.strip()

def main():
    parser = argparse.ArgumentParser(description="Few-shot Optimize STT Prompts based on Error Analysis")
    parser.add_argument("--summary_csv", default="execution_log/evaluation_summary.csv", help="Path to evaluation summary CSV")
    parser.add_argument("--meta_prompt", default="prompt/meta_prompt.txt", help="Path to meta prompt template")
    parser.add_argument("--prompt_dir", default="prompt", help="Directory containing original prompts")
    parser.add_argument("--output_dir", default="prompt/prompt-modified", help="Directory to save modified prompts")
    parser.add_argument("--critic_dir", default="execution_log/critic_results", help="Directory containing critic results")
    parser.add_argument("--project_id", type=str, default="my-argolis-prj", help="GCP Project ID")
    parser.add_argument("--location", type=str, default="global", help="GCP Location")
    parser.add_argument("--top_n", type=int, default=15, help="Number of worst examples to analyze")
    parser.add_argument("--model", type=str, default="gemini-3.1-pro-preview", help="Model to use for generating optimized prompts")
    
    args = parser.parse_args()
    
    # Get absolute path based on the working directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    summary_csv_path = os.path.join(base_dir, args.summary_csv)
    meta_prompt_path = os.path.join(base_dir, args.meta_prompt)
    prompt_dir_path = os.path.join(base_dir, args.prompt_dir)
    output_dir_path = os.path.join(base_dir, args.output_dir)
    
    os.makedirs(output_dir_path, exist_ok=True)
    
    if not os.path.exists(summary_csv_path):
        print(f"Error: Cannot find Summary CSV file: {summary_csv_path}")
        return

    df_summary = pd.read_csv(summary_csv_path)
    
    # 1. Filter for group 'train' only
    df_train = df_summary[df_summary['group'] == 'train']
    
    if df_train.empty:
        print("Warning: No data found for group 'train' in summary CSV.")
        return

    # Remove duplicates based on lang, group, model, config, and prompt_version (keep the latest)
    subset_keys = ['lang', 'group', 'model', 'config']
    if 'prompt_version' in df_train.columns:
        subset_keys.append('prompt_version')
    df_train = df_train.drop_duplicates(subset=subset_keys, keep='last')

    if not os.path.exists(meta_prompt_path):
        print(f"Error: Cannot find Meta Prompt file: {meta_prompt_path}")
        return

    with open(meta_prompt_path, 'r', encoding='utf-8') as f:
        meta_prompt_template = f.read()
        
    client = genai.Client(vertexai=True, project=args.project_id, location=args.location)
    
    for _, row in df_train.iterrows():
        lang = row['lang']
        group = row['group']
        model_used = row['model']
        config_used = row['config']
        prompt_version = row.get('prompt_version', '')
        pv_clean = str(prompt_version).replace('.txt', '') if pd.notna(prompt_version) and prompt_version else 'default'
        local_csv = row['local_inference_csv_path']
        
        print(f"[{group}_{lang}] Processing Model: {model_used}, Config: {config_used}...")
        
        # Handle relative paths
        if not os.path.isabs(local_csv):
            local_csv = os.path.join(base_dir, local_csv)
            
        if not os.path.exists(local_csv):
            print(f"  -> File not found, skipping: {local_csv}")
            continue
            
        print(f"  -> Loading evaluation data and calculating WER/CER...")
        df_local = pd.read_csv(local_csv)
        df_local = calculate_errors(df_local)
        
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
            
        critic_csv = os.path.join(base_dir, args.critic_dir, f"critic_{group}_{lang}_{model_used}_{config_used}_{pv_clean}.csv")
        critiques_text = ""
        if os.path.exists(critic_csv):
            df_critics = pd.read_csv(critic_csv)
            for j, (_, critic_row) in enumerate(df_critics.iterrows(), 1):
                critiques_text += f"## Critique {j}\n"
                critiques_text += f"Error Pattern: {critic_row.get('error_pattern', '')}\n"
                critiques_text += f"Affected Cases: {critic_row.get('affected_cases', '')}\n"
                critiques_text += f"Reason for Failure: {critic_row.get('reason_for_failure', '')}\n"
                critiques_text += f"Generalizable Improvement: {critic_row.get('generalizable_improvement', '')}\n\n"
        else:
            print(f"  -> Warning: Critic file ({critic_csv}) not found.")

        # Read the previously used prompt
        prompt_path = os.path.join(prompt_dir_path, str(prompt_version)) if prompt_version else os.path.join(prompt_dir_path, "prompt_v2.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                used_prompt = f.read()
        else:
            fallback_path = os.path.join(prompt_dir_path, "prompt_v2.txt")
            if os.path.exists(fallback_path):
                with open(fallback_path, 'r', encoding='utf-8') as f:
                    used_prompt = f.read()
            else:
                used_prompt = "Default STT Prompt (Not Found)"
                print(f"  -> Warning: Previous prompt file not found.")
            
        # 2. Insert data into meta prompt template
        analysis_prompt = meta_prompt_template.replace('{args.model}', str(model_used))\
                                              .replace('{used_config}', str(config_used))\
                                              .replace('{used_prompt}', used_prompt)\
                                              .replace('{error_cases_text}', error_cases_text)\
                                              .replace('{critiques_text}', critiques_text)
                                              
        # Additional instructions for easy prompt extraction
        analysis_prompt += "\n\nIMPORTANT: Please provide the optimized prompt enclosed in a ```prompt ... ``` block."
        
        print(f"  -> Generating optimized prompt using LLM ({args.model})...")
        try:
            response = client.models.generate_content(
                model=args.model,
                contents=analysis_prompt,
            )
            
            # 3. Save Improved Prompt
            new_prompt = extract_prompt_from_response(response.text)
            
            output_file = os.path.join(output_dir_path, f"prompt_optimized_{group}_{lang}_{model_used}_{config_used}_{pv_clean}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(new_prompt)
                
            report_file = os.path.join(output_dir_path, f"analysis_report_optimized_{group}_{lang}_{model_used}_{config_used}_{pv_clean}.md")
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(response.text)
                
            print(f"  -> ✅ Completed: Improved prompt saved ({output_file})")
            
        except Exception as e:
            print(f"  -> ❌ Error occurred ({lang}): {e}")

if __name__ == "__main__":
    main()