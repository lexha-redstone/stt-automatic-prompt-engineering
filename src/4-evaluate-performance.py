import pandas as pd
import json
import os
import glob
from google.cloud import storage
import jiwer

# ==========================================
# 0. Configs
# ==========================================
# Dynamically determines the current location of the script (project root).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_EXTERNAL = os.path.abspath(os.path.join(BASE_DIR, "../../../data"))

LABEL_CSV_PATH = os.path.join(DATA_DIR_EXTERNAL, "fleurs/total_df_processed_with_gcs.csv")
SILENCE_CSV_PATH = os.path.join(DATA_DIR_EXTERNAL, "fleurs/total_df_silence_processed_with_gcs.csv")

LOG_DIR = os.path.join(BASE_DIR, "execution_log")
EVAL_SUMMARY_CSV = os.path.join(LOG_DIR, "evaluation_summary.csv")
BATCH_RESULTS_TEMP_DIR = os.path.join(BASE_DIR, "batch_results")

# Optional alias for backward compatibility in the script
SCRIPT_DIR = BASE_DIR

def download_gcs_folder(gcs_uri, local_dir):
    """Downloading .jsonl from GCS URI"""
    if not gcs_uri.startswith("gs://"):
        raise ValueError("GCS URI must start with gs://")
    
    storage_client = storage.Client()
    path_parts = gcs_uri[5:].split("/", 1)
    bucket_name = path_parts[0]
    prefix = path_parts[1] if len(path_parts) > 1 else ""
    
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    os.makedirs(local_dir, exist_ok=True)
    downloaded_files = []
    
    for blob in blobs:
        if blob.name.endswith(".jsonl"):
            local_file_path = os.path.join(local_dir, os.path.basename(blob.name))
            print(f"Downloading: gs://{bucket_name}/{blob.name} -> {local_file_path}")
            blob.download_to_filename(local_file_path)
            downloaded_files.append(local_file_path)
            
    return downloaded_files

def main():
    if not os.path.exists(LABEL_CSV_PATH):
        print(f"Error: Dataset {LABEL_CSV_PATH} file not found.")
        return
        
    # 2. Load the entire label dataset
    print(f"Loading original dataset...")
    total_df_reg = pd.read_csv(LABEL_CSV_PATH)
    total_df_silence = pd.read_csv(SILENCE_CSV_PATH) if os.path.exists(SILENCE_CSV_PATH) else None
    
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
    ])
    
    eval_summary_csv = EVAL_SUMMARY_CSV
    processed_jobs = set()
    if os.path.exists(eval_summary_csv):
        try:
            eval_df = pd.read_csv(eval_summary_csv)
            if "local_inference_csv_path" in eval_df.columns:
                processed_jobs = set(eval_df["local_inference_csv_path"].dropna().tolist())
        except Exception as e:
            print(f"Loading Error : eval summary csv: {e}")

    log_files = glob.glob(os.path.join(LOG_DIR, "*.json"))
    if not log_files:
        print(f"Error: No log files found in {LOG_DIR}.")
        return

    for log_file in log_files:
        print(f"\nProcessing log file: {log_file}")
        with open(log_file, "r", encoding="utf-8") as f:
            try:
                jobs_log = json.load(f)
            except json.JSONDecodeError:
                print(f"JSON Parsing Error: {log_file}")
                continue
            
        for job in jobs_log:
            print("\n" + "="*50)
            print(f"Job evaluation start: {job.get('splitter')} / {job.get('model')} / {job.get('config_name')}")
            
            gcs_output_base_uri = job.get("gcs_output_base_uri")
            local_output_csv = job.get("local_output_csv")
            prompt_version = job.get("prompt_version", "")
            
            if prompt_version and local_output_csv and local_output_csv.endswith(".csv"):
                pv_clean = prompt_version.replace(".txt", "")
                import re
                base_csv = re.sub(r'_prompt_v[^\.]+\.csv$', '.csv', local_output_csv)
                if not base_csv.endswith(f"_{pv_clean}.csv"):
                    local_output_csv = f"{base_csv[:-4]}_{pv_clean}.csv"
                else:
                    local_output_csv = base_csv
            
            if not gcs_output_base_uri or not local_output_csv:
                print("Required key (gcs_output_base_uri, local_output_csv) is missing, skipping.")
                continue

            if local_output_csv in processed_jobs:
                print(f"Already processed (local_inference_csv_path: {local_output_csv}). Skipping.")
                continue
                
            # Temp directory
            pv_suffix = f"_{prompt_version.replace('.txt', '')}" if prompt_version else ""
            local_download_dir = os.path.join(BATCH_RESULTS_TEMP_DIR, f"temp_{job.get('splitter')}_{job.get('model')}_{job.get('config_name')}{pv_suffix}")
            
            print(f"Downloading results from GCS: {gcs_output_base_uri}")
            jsonl_files = download_gcs_folder(gcs_output_base_uri, local_download_dir)
            
            if not jsonl_files:
                print("No files downloaded. Skipping.")
                continue
                
            print("Parsing downloaded JSONL files...")
            predictions_map = {}
            
            for file_path in jsonl_files:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        data = json.loads(line)
                        
                        file_uri = None
                        try:
                            contents = data.get("request", {}).get("contents", [])
                            if contents:
                                parts = contents[0].get("parts", [])
                                if parts:
                                    file_uri = parts[0].get("fileData", {}).get("fileUri", "")
                        except Exception:
                            pass
                            
                        if not file_uri:
                            continue
                            
                        pred_text = ""
                        try:
                            candidates = data.get("response", {}).get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                if parts:
                                    pred_text = parts[0].get("text", "")
                        except Exception as e:
                            print(f"Prediction parsing error ({file_uri}): {e}")
                            
                        predictions_map[file_uri] = pred_text
                        
            print(f"Successfully parsed {len(predictions_map)} predictions.")
            
            # Filter data corresponding to the current job
            lang = job.get("lang")
            group = job.get("group")
            
            if group == 'silence':
                if total_df_silence is None:
                    print(f"Warning: {SILENCE_CSV_PATH} file not found, cannot proceed with Silence evaluation.")
                    continue
                job_df = total_df_silence[(total_df_silence['lang'] == lang)].copy()
            else:
                job_df = total_df_reg[(total_df_reg['lang'] == lang) & (total_df_reg['group_name'] == group)].copy()
            
            print(f"Number of data to evaluate: {len(job_df)}")
            
            job_df["prediction"] = ""
            
            # Prediction mapping
            for idx, row in job_df.iterrows():
                gcs_uri = row["gcs_uri"]
                if gcs_uri in predictions_map:
                    job_df.loc[idx, "prediction"] = predictions_map[gcs_uri]
                    
            # Handling missing values before evaluation
            job_df["prediction"] = job_df["prediction"].fillna("")
            job_df["transcription"] = job_df["transcription"].fillna("")
            
            valid_refs = []
            valid_hyps = []
            
            for _, row in job_df.iterrows():
                r = row["transcription"]
                h = row["prediction"]
                
                if not isinstance(r, str) or not r.strip():
                    continue
                    
                r_trans = transformation(r)
                h_trans = transformation(h)
                
                if not r_trans.strip():
                    continue
                if not h_trans.strip():
                    h_trans = " " 
                    
                valid_refs.append(r_trans)
                valid_hyps.append(h_trans)
                
            if not valid_refs:
                print("No valid samples to evaluate.")
                continue
                
            wer = jiwer.wer(valid_refs, valid_hyps)
            cer = jiwer.cer(valid_refs, valid_hyps)
            
            print(f"Total Samples evaluated: {len(valid_refs)}")
            print(f"WER (Word Error Rate): {wer * 100:.2f}%")
            print(f"CER (Character Error Rate): {cer * 100:.2f}%")
            
            # Save results
            os.makedirs(os.path.dirname(local_output_csv), exist_ok=True)
            job_df.to_csv(local_output_csv, index=False, encoding="utf-8-sig")
            print(f"Result saved: {local_output_csv}")

            # Save cumulative evaluation results
            eval_log_exists = os.path.exists(eval_summary_csv)
            
            eval_record = {
                "LOG_FILE": log_file,
                "lang": lang,
                "group": group,
                "model": job.get("model", ""),
                "local_inference_csv_path": local_output_csv,
                "config": job.get("config_name", ""),
                "prompt_version": job.get("prompt_version", ""),
                "wer": wer,
                "cer": cer
            }
            eval_df_to_save = pd.DataFrame([eval_record])
            eval_df_to_save.to_csv(eval_summary_csv, mode='a', header=not eval_log_exists, index=False, encoding="utf-8-sig")
            print(f"Evaluation log added: {eval_summary_csv}")
            
            # Add the processed job to processed_jobs to prevent duplicates
            processed_jobs.add(local_output_csv)

if __name__ == "__main__":
    main()
