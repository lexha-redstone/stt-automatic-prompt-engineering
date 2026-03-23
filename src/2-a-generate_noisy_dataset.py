import pandas as pd
import os
import random
import subprocess
import math
from pydub import AudioSegment
from pydub.effects import speedup

# ==========================================
# 0. Configs
# ==========================================
# Dynamically determines the current location of the script (project root).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../data"))

INPUT_CSV_PATH = os.path.join(DATA_DIR, "fleurs/total_df_medical_yn.csv")
OUTPUT_CSV_PATH = os.path.join(DATA_DIR, "fleurs/total_df_processed.csv")
OUTPUT_AUDIO_DIR = os.path.join(DATA_DIR, "noisy-stt/noisy-synthesized/total")
NOISE_DIR = os.path.join(DATA_DIR, "noisy-stt/FSDnoisy18k.audio_test")

TEMP_MAIN_WAV = os.path.join(BASE_DIR, "temp_main.wav")
TEMP_NOISE_WAV = os.path.join(BASE_DIR, "temp_noise.wav")

def convert_to_16bit_wav(input_file, output_file):
    """Force-convert any audio to standard 16-bit WAV format using FFmpeg."""
    command = [
        "ffmpeg", "-y",
        "-i", input_file,
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        output_file
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_file

def process_audio(main_file, noise_file, output_file, speed_factor=1.0, noise_multiplier=0.3):
    """
    noise_multiplier: Ratio of noise volume to main audio volume.
    """
    print(f"1. Converting audio format (16-bit standardization)... {main_file}")
    clean_main = convert_to_16bit_wav(main_file, TEMP_MAIN_WAV)
    clean_noise = convert_to_16bit_wav(noise_file, TEMP_NOISE_WAV)

    print("2. Loading converted audio...")
    main_audio = AudioSegment.from_file(clean_main)
    noise_audio = AudioSegment.from_file(clean_noise)

    print(f"3. Converting main audio speed to {speed_factor}x...")
    if speed_factor != 1.0:
        fast_main_audio = speedup(main_audio, playback_speed=speed_factor)
    else:
        fast_main_audio = main_audio

    print("4. Analyzing background noise volume and automatically matching...")
    main_dbfs = fast_main_audio.dBFS
    noise_dbfs = noise_audio.dBFS

    db_adjustment = main_dbfs - noise_dbfs
    normalized_noise = noise_audio + db_adjustment

    if noise_multiplier <= 0:
        final_noise = normalized_noise - 100 
    else:
        volume_change_db = 20 * math.log10(noise_multiplier)
        final_noise = normalized_noise + volume_change_db

    print(f"   -> Applied volume multiplier: {noise_multiplier}x (approx {volume_change_db:.2f} dB adjustment)")

    print("5. Matching noise length and synthesizing audio...")
    required_length = len(fast_main_audio)
    
    if len(final_noise) < required_length:
        loop_count = (required_length // len(final_noise)) + 1
        final_noise = final_noise * loop_count
    
    final_noise = final_noise[:required_length]

    final_audio = fast_main_audio.overlay(final_noise)

    print("6. Saving final result...")
    # The original notebook saved as mp3, but the problem specifies a .wav extension, so save as wav.
    final_audio.export(output_file, format="wav")
    
    os.remove(clean_main)
    os.remove(clean_noise)
    print(f"✨ Task Completed! Check {output_file}.\n")

def generate_noisy_dataset(input_csv_path, output_csv_path, output_audio_dir):
    # 1. Read the CSV file provided by the input path into a pandas dataframe.
    df = pd.read_csv(input_csv_path)
    df = df[df['medical_yn'] == True].reset_index(drop=True)
    
    noise_dir = NOISE_DIR
    noise_files = [f for f in os.listdir(noise_dir) if f.endswith('.wav')]
    
    # Pre-specified path to save synthesized audio
    os.makedirs(output_audio_dir, exist_ok=True)
    
    successful_rows = []
    
    # 2. Iterate through the pandas dataframe you just read, grouped by language, until 100 samples are processed.
    for lang, group in df.groupby('lang'):
        success_count = 0
        for index, row in group.iterrows():
            if success_count >= 100:
                break
                
            # 3. Read the audio file from the path in each row and consider it as the main file.
            main_file = row.get('file_path', row.get('audio'))
            row_id = row['id']
            
            # Split each language into 50 for train and 50 for test
            group_name = 'train' if success_count < 50 else 'test'
            
            # 4. Randomly select one noise file from the specified path and consider it as the noise file.
            noise_filename = random.choice(noise_files)
            noise_file = os.path.join(noise_dir, noise_filename)
            
            # Record which noise file was read (the number excluding the extension from the filename)
            noise_id = os.path.splitext(noise_filename)[0]
            
            # 6. Create the filename as f"{lang}-{group_name}-{index}-{row_id}-{noise_id}.wav".
            output_filename = f"{lang}-{group_name}-{index}-{row_id}-{noise_id}.wav"
            output_file = os.path.join(output_audio_dir, output_filename)
            
            # 5. Use the function to synthesize main_file and noise_file. Handle exceptions and skip if failed.
            print(f"[{success_count + 1}/100 for {lang}] Processing id: {row_id}, noise_id: {noise_id}")
            try:
                process_audio(main_file, noise_file, output_file, speed_factor=1.0, noise_multiplier=0.3)
                
                # If successful, update the row data and add it to the list.
                new_row = row.copy()
                new_row['group_name'] = group_name
                new_row['noisy_output_path'] = output_file
                new_row['noise_id'] = noise_id
                successful_rows.append(new_row)
                
                success_count += 1
            except Exception as e:
                print(f"Failed to process id {row_id} (lang: {lang}): {e}. Skipping to next data...")
                continue
                
        if success_count < 100:
            print(f"Warning: Only generated {success_count} for {lang} due to lack of available valid data.")
        
    # 8. Now create an updated pandas dataframe and output it as a .csv file to the pre-specified path.
    final_df = pd.DataFrame(successful_rows)
    final_df.to_csv(output_csv_path, index=False)
    print(f"Updated dataframe saved to {output_csv_path}")
    
    return final_df

if __name__ == "__main__":
    # Example usage (You can run this script by inserting the CSV you want here)
    _ = generate_noisy_dataset(INPUT_CSV_PATH, OUTPUT_CSV_PATH, OUTPUT_AUDIO_DIR)
