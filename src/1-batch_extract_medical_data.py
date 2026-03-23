import pandas as pd
import json
import os
import time
from google.cloud import storage
from google import genai
from google.genai.types import CreateBatchJobConfig
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 0. Settings / Configs (Action required: Update according to your environment)
# ==========================================
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
BUCKET_NAME = os.getenv("BUCKET_NAME")

MODEL_NAME = "gemini-3.1-pro-preview"
CHUNK_SIZE = 50

# Dynamically determines the current location of the script (project root).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The data directory is located at .../work/data (outside the project folder)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../data"))

INPUT_CSV_PATH = os.path.join(DATA_DIR, "fleurs/total_df.csv")
OUTPUT_CSV_PATH = os.path.join(DATA_DIR, "fleurs/total_df_medical_yn.csv")

LOCAL_JSONL_DIR = os.path.join(BASE_DIR, "data")
LOCAL_DOWNLOAD_DIR = os.path.join(BASE_DIR, "batch_results")

# Dynamic naming based on time
JOB_TAG = f"medical-ext-{int(time.time())}"
JSONL_FILENAME = f"requests_{JOB_TAG}.jsonl"
JSONL_LOCAL_PATH = os.path.join(LOCAL_JSONL_DIR, JSONL_FILENAME)

# GCS Path Configuration
GCS_INPUT_DIR = "batch_input/medical_extract"
GCS_JSONL_PATH = f"gs://{BUCKET_NAME}/{GCS_INPUT_DIR}/{JSONL_FILENAME}"
GCS_OUTPUT_BASE_URI = f"gs://{BUCKET_NAME}/batch_output/medical_extract_{JOB_TAG}"


def download_gcs_folder(gcs_uri, local_dir, project_id):
    """Downloads results from the GCS URI to the local machine."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError("GCS URI must start with gs://")
    
    storage_client = storage.Client(project=project_id)
    path_parts = gcs_uri[5:].split("/", 1)
    bucket_name = path_parts[0]
    prefix = path_parts[1] if len(path_parts) > 1 else ""
    
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    os.makedirs(local_dir, exist_ok=True)
    downloaded_files = []
    
    file_found = False
    for blob in blobs:
        if blob.name.endswith(".jsonl"):
            file_found = True
            local_file_path = os.path.join(local_dir, os.path.basename(blob.name))
            print(f"Downloading: gs://{bucket_name}/{blob.name} -> {local_file_path}")
            blob.download_to_filename(local_file_path)
            downloaded_files.append(local_file_path)
            
    if not file_found:
        print(f"Warning: No .jsonl files found at {gcs_uri}.")
            
    return downloaded_files


def process_results(df, jsonl_files):
    print("\nParsing downloaded JSONL files and merging the results...")
    
    # chunk_idx -> list of booleans mapping
    results_map = {}
    
    for file_path in jsonl_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                
                chunk_idx = None
                
                # key parsing ("req-{chunk_idx}" format)
                key = data.get("key", "")
                if isinstance(key, str) and key.startswith("req-"):
                    try:
                        chunk_idx = int(key.split("-")[1])
                    except ValueError:
                        pass
                
                if chunk_idx is None:
                    continue
                
                try:
                    candidates = data.get("response", {}).get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            pred_text = parts[0].get("text", "")
                            
                            # Remove markdown code blocks before JSON parsing
                            pred_text = pred_text.strip()
                            if pred_text.startswith("```json"):
                                pred_text = pred_text[7:]
                            elif pred_text.startswith("```"):
                                pred_text = pred_text[3:]
                            if pred_text.endswith("```"):
                                pred_text = pred_text[:-3]
                            pred_text = pred_text.strip()
                            
                            boolean_list = json.loads(pred_text)
                            results_map[chunk_idx] = boolean_list
                except Exception as e:
                    print(f"[Parsing Error] Chunk Index {chunk_idx}: {e}\n Original Text: {pred_text}")
                    results_map[chunk_idx] = []

    # Add medical_yn column to the entire DataFrame
    medical_yn_column = [False] * len(df) # Default value False if parsing fails or is missing
    
    for i in range(0, len(df), CHUNK_SIZE):
        chunk_idx = i // CHUNK_SIZE
        end_idx = min(i + CHUNK_SIZE, len(df))
        chunk_len = end_idx - i
        
        bool_list = results_map.get(chunk_idx, [])
        
        # Validating returned list length and handling exceptions
        if len(bool_list) != chunk_len:
            print(f"[Warning] Chunk {chunk_idx}: The number of input texts ({chunk_len}) and the number of results ({len(bool_list)}) do not match. Missing values are filled with False.")
            while len(bool_list) < chunk_len:
                bool_list.append(False)
            bool_list = bool_list[:chunk_len]
            
        for j in range(chunk_len):
            medical_yn_column[i + j] = bool_list[j]
            
    df["medical_yn"] = medical_yn_column
    
    # Save the final result with index=False
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n✅ Final result saved to: {OUTPUT_CSV_PATH}")


def main():
    # 1. Reading the original CSV file
    print(f"[1/7] Reading the original CSV file: {INPUT_CSV_PATH}")
    df = pd.read_csv(INPUT_CSV_PATH)
    
    os.makedirs(os.path.dirname(JSONL_LOCAL_PATH), exist_ok=True)
    
    # 2. Splitting data into chunks of CHUNK_SIZE and creating a JSONL file
    print(f"[2/7] Splitting data into chunks of {CHUNK_SIZE} and creating a JSONL file: {JSONL_LOCAL_PATH}")
    with open(JSONL_LOCAL_PATH, "w", encoding="utf-8") as jsonl_file:
        for i in range(0, len(df), CHUNK_SIZE):
            chunk_idx = i // CHUNK_SIZE
            end_idx = min(i + CHUNK_SIZE, len(df))
            
            chunk_transcriptions = df["transcription"].iloc[i:end_idx].fillna("").tolist()
            
            prompt_text = (
                "You are an expert text classifier. I will give you a list of transcriptions.\n"
                "For each transcription, determine if it belongs to the medical or healthcare domain.\n"
                "Return the result STRICTLY as a JSON array of booleans, where 'true' means it is medical, "
                "and 'false' means it is not. The length of the array must be exactly the same as the number of transcriptions.\n\n"
                "Transcriptions:\n"
            )
            for idx, text in enumerate(chunk_transcriptions):
                prompt_text += f"[{idx}] {text}\n"
                
            request_obj = {
                "key": f"req-{chunk_idx}",
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": prompt_text
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "response_mime_type": "application/json",
                    }
                }
            }
            jsonl_file.write(json.dumps(request_obj) + "\n")

    # 3. Uploading the JSONL file to GCS
    print(f"[3/7] Uploading the generated JSONL file to GCS: {GCS_JSONL_PATH}")
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"{GCS_INPUT_DIR}/{JSONL_FILENAME}")
    blob.upload_from_filename(JSONL_LOCAL_PATH)

    # 4. Starting Batch Inference
    print(f"[4/7] Starting Google GenAI Batch Job. (Model: {MODEL_NAME})")
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    try:
        job = client.batches.create(
            model=MODEL_NAME,
            src=GCS_JSONL_PATH,
            config=CreateBatchJobConfig(dest=GCS_OUTPUT_BASE_URI),
        )

        print(f"\n✅ Batch Inference job submitted: {job.name}")
        
        # 5. Waiting for Batch Job completion
        print(f"[5/7] Waiting for Batch Job completion... (Job Name: {job.name})")
        
        while True:
            job_status = client.batches.get(name=job.name)
            state = getattr(job_status, 'state', str(job_status))
            
            if "SUCCEEDED" in state and "PARTIALLY" not in state:
                print("\n✅ Batch Job completed successfully!")
                break
            elif "FAILED" in state or "CANCELLED" in state or "PARTIALLY_SUCCEEDED" in state:
                print(f"\n❌ Batch Job failed or partially succeeded. Status: {state}")
                if "PARTIALLY_SUCCEEDED" in state:
                    print("Partially succeeded, attempting to parse.")
                    break
                else:
                    return # Failure
            else:
                print(f"Current status: {state}... Checking again in 30 seconds.")
                time.sleep(30)
                
        # 6. Downloading results from GCS
        print(f"\n[6/7] Downloading results from GCS.")
        final_output_uri = getattr(job_status, 'output_uri', GCS_OUTPUT_BASE_URI)
        jsonl_files = download_gcs_folder(final_output_uri, LOCAL_DOWNLOAD_DIR, PROJECT_ID)
        
        if not jsonl_files:
            print("No results downloaded.")
            return
            
        # 7. Parsing results and adding them to the CSV
        print(f"\n[7/7] Parsing results and adding them to the CSV.")
        process_results(df, jsonl_files)

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
