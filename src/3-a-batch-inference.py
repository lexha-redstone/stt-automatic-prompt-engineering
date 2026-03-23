import pandas as pd
import json
import os
import argparse
from datetime import datetime
from google.cloud import storage
from google import genai
from google.genai.types import CreateBatchJobConfig
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 0. Settings / Configs (Modify according to user environment)
# ==========================================
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Dynamically determines the current location of the script (project root).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_EXTERNAL = os.path.abspath(os.path.join(BASE_DIR, "../../../data"))

BASE_CSV_PATH = os.path.join(DATA_DIR_EXTERNAL, "fleurs/total_df_processed_with_gcs.csv")

PROMPT_VERSION = "v3"  # "v2" / "v3"
print(f"###PROMPT VERSION### : {PROMPT_VERSION}")
PROMPT_DIR = os.path.join(BASE_DIR, "prompt", "prompt-modified")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOCAL_RESULTS_DIR = os.path.join(BASE_DIR, "batch_inference_results")
EXECUTION_LOG_DIR = os.path.join(BASE_DIR, "execution_log")

GCS_INPUT_DIR_PREFIX = "batch_input"
GCS_OUTPUT_DIR_PREFIX = "batch_output"

# 1. Combinations
LANGUAGES = [
    {"lang": "en_us", "code": "en_us"},
    {"lang": "ar_eg", "code": "ar_eg"},
    {"lang": "cmn_hans_cn", "code": "cmn_hans_cn"},
    {"lang": "el_gr", "code": "el_gr"},
    {"lang": "he_il", "code": "he_il"},
    {"lang": "hi_in", "code": "hi_in"},
    {"lang": "ja_jp", "code": "ja_jp"},
    {"lang": "ko_kr", "code": "ko_kr"},
    {"lang": "yue_hant_hk", "code": "yue_hant_hk"},

]

# "train"
GROUPS = [
    "test"
]
MODELS = [
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
]
#     "gemini-2.5-flash",
    # "gemini-3-flash-preview",
    # "gemini-3.1-pro-preview",



CONFIGS = [
    {"name": "default", "generationConfig": None},
    {"name": "temperature", "generationConfig": {"audio_timestamp": True, "temperature": 0.2}},
    {"name": "audio_ts_true", "generationConfig": {"audio_timestamp": True, "topK": 1, "topP": 0.1}},

]


def main():
    parser = argparse.ArgumentParser(description="Vertex AI Batch Inference - Multiple Combinations")
    args = parser.parse_args()

    # Initialize GCS and GenAI Client
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    # Create output folders
    os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
    os.makedirs(EXECUTION_LOG_DIR, exist_ok=True)

    # 2. Read dataset
    job_records = []
    
    print(f"Reading base dataset: {BASE_CSV_PATH}")
    base_df = pd.read_csv(BASE_CSV_PATH)

    # 3. Loop for submitting Batch Inference for each combination
    for lang_info in LANGUAGES:
        # --- Load prompt ---
        lang = lang_info['lang']
        if PROMPT_VERSION == "v3":
            current_prompt_path = os.path.join(PROMPT_DIR, f"prompt_v3_{lang}.txt")
        else:
            current_prompt_path = os.path.join(BASE_DIR, "prompt", "prompt_v2.txt")
        
        if os.path.exists(current_prompt_path):
            print(f"\nReading Prompt: {current_prompt_path}")
            with open(current_prompt_path, "r", encoding="utf-8") as f:
                prompt_template = f.read()
            prompt_version = os.path.basename(current_prompt_path)
        else:
            print(f"\nWarning: {current_prompt_path} file not found.")
            raise FileNotFoundError(f"{current_prompt_path} not found")
        # ---------------------------

        for group in GROUPS:
            splitter = f"{group}_{lang_info['lang']}"
            
            print(f"\n==============================================")
            print(f"[{splitter}] Filtering dataset...")
            df = base_df[(base_df['lang'] == lang_info['lang']) & (base_df['group_name'] == group)]
            print(f"Target Data size : {df.shape}")

            if df.empty:
                print(f"\nSkipped: {splitter} data not exists.")
                continue

            for model in MODELS:
                for config_info in CONFIGS:
                    config_name = config_info["name"]
                    generation_config = config_info["generationConfig"]
                    tag = f"{model}-{config_name}"
                    
                    print(f"\n>>> Model: {model} | Config: {config_name}")

                    # Remove extension from prompt_version (e.g., prompt_v3_en_us.txt -> prompt_v3_en_us)
                    prompt_version_slug = os.path.splitext(prompt_version)[0]
                    
                    jsonl_filename = f"{splitter}_{tag}_{prompt_version_slug}_requests.jsonl"
                    jsonl_local_path = os.path.join(DATA_DIR, jsonl_filename)
                    
                    gcs_input_dir = f"{GCS_INPUT_DIR_PREFIX}/{splitter}"
                    gcs_jsonl_path = f"gs://{BUCKET_NAME}/{gcs_input_dir}/{jsonl_filename}"
                    
                    gcs_output_base_uri = f"gs://{BUCKET_NAME}/{GCS_OUTPUT_DIR_PREFIX}/{splitter}_{tag}_{prompt_version_slug}"

                    prompt_text = prompt_template.format(language_code=lang_info["code"])

                    # (1) Create JSONL file and (optional) upload audio
                    with open(jsonl_local_path, "w", encoding="utf-8") as jsonl_file:
                        for index, row in df.iterrows():
                            gcs_audio_uri = row.get("gcs_uri")
                            if pd.isna(gcs_audio_uri):
                                continue

                            # Audio file upload logic is separated into 2-c-upload_audio.py.

                            request_obj = {
                                "key": f"req-{index}-{prompt_version_slug}",
                                "request": {
                                    "contents": [
                                        {
                                            "role": "user",
                                            "parts": [
                                                {"fileData": {"mimeType": "audio/wav", "fileUri": gcs_audio_uri}},
                                                {"text": prompt_text}
                                            ]
                                        }
                                    ]
                                }
                            }
                            if generation_config:
                                request_obj["request"]["generationConfig"] = generation_config
                            jsonl_file.write(json.dumps(request_obj) + "\n")

                    # (2) Upload created JSONL file to GCS
                    print(f"  - Uploading JSONL: {gcs_jsonl_path}")
                    blob = bucket.blob(f"{gcs_input_dir}/{jsonl_filename}")
                    blob.upload_from_filename(jsonl_local_path)

                    # (3) Submit Batch Inference job
                    print("  - Submitting Batch Job...")
                    try:
                        job = client.batches.create(
                            model=model,
                            src=gcs_jsonl_path,
                            config=CreateBatchJobConfig(dest=gcs_output_base_uri),
                        )
                        print(f"  ✅ Submitted successfully: {job.name}")
                        
                        # Append dictionary for saving execution records
                        job_records.append({
                            "lang": lang_info["lang"],
                            "group": group,
                            "splitter": splitter,
                            "model": model,
                            "config_name": config_name,
                            "job_name": job.name,
                            "prompt_version": prompt_version,
                            "gcs_output_base_uri": gcs_output_base_uri,
                            "local_output_csv": os.path.join(LOCAL_RESULTS_DIR, f"evaluation_{splitter}_{tag}_{prompt_version_slug}.csv"),
                            "submitted_at": datetime.now().isoformat()
                        })
                    except Exception as e:
                        print(f"  ❌ Failed to submit job: {e}")

    # 4. Save Batch Job execution records (for the next evaluation script)
    if job_records:
        log_filename = f"batch_jobs_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        log_filepath = os.path.join(EXECUTION_LOG_DIR, log_filename)
        
        with open(log_filepath, "w", encoding="utf-8") as f:
            json.dump(job_records, f, indent=4, ensure_ascii=False)
            
        print(f"\n==============================================")
        print(f"🎉 All Batch Jobs submitted successfully.")
        print(f"Execution records (Job Info) saved to:\n -> {log_filepath}")
        print(f"The next evaluation script can read this JSON file to download results from GCS URI and proceed with evaluation.")
    else:
        print("\n⚠️ No Batch Jobs were submitted.")

if __name__ == "__main__":
    main()