# STT Prompt Engineering Pipeline

This repository automates the prompt engineering and evaluation lifecycle for Speech-to-Text (STT) use cases using large language models. It includes optional modules for synthetic data generation and provides an end-to-end framework spanning inference, evaluation, critic-based optimization, and reporting.

## 🌟 Key Features
- **Data Preparation (Optional)**: Generate noisy datasets or add artificial silence to evaluate STT robustness.
- **Batch Inference**: Distributed, asynchronous processing of audio datasets using Vertex AI Batch jobs.
- **Automated Evaluation**: Quantitatively assess the performance of the model on the generated inferences.
- **Critic-based Optimization**: Automatically analyze failures and identify optimization opportunities for your prompts.
- **Few-Shot Optimization**: Dynamically generate new, optimized prompts incorporating error-based few-shot examples.
- **Report Generation**: Render detailed performance visualizations and transcripts directly to HTML for easy review.

## 🛠 Prerequisites

1. **Python 3.13+**
2. **Google Cloud Project** with Vertex AI API enabled.
3. Authenticate to Google Cloud using your environment (e.g., `gcloud auth application-default login`).
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you have `google-genai`, `python-dotenv`, and other necessary libraries installed).*

5. Create a `.env` file in the project root:
   ```env
   PROJECT_ID=your-google-cloud-project-id
   LOCATION=your-gcp-region (e.g., us-central1)
   ```

## 🚀 How to Run the Pipeline

> [!IMPORTANT]
> **Data Preparation is Required**: Before running the automation pipeline, you must ensure your evaluation datasets are prepared and uploaded. The pipeline script (`run_pipeline.py`) assumes that the inference scripts (`3-a` and `3-b`) have valid audio URI lists to process. 
> 
> If you are starting from scratch, please follow the **Data Preparation** steps below first.

The easiest way to execute the full evaluation and prompt engineering cycle is using the `run_pipeline.py` script. This script acts as an orchestrator and automatically waits for asynchronous batch inference jobs to complete before moving on to the next steps.

```bash
python run_pipeline.py
```

### 📂 Pipeline Steps Overview (`src/`)

The pipeline comprises several sequential scripts in the `src/` directory. You can run them individually if you want granular control.

#### 1. Data Preparation (Optional)
If you do not have a pre-existing dataset, you can utilize the dataset generation scripts:
- `0-access-dataset.ipynb`: Notebook for initial dataset exploration.
- `1-batch_extract_medical_data.py`: Extract domain-specific (e.g., medical) STT evaluation data.
- `2-a-generate_noisy_dataset.py`: Artificially introduce noise into clean audio to test robustness.
- `2-b-generate_silence_dataset.py`: Inject silence blocks into audio to evaluate hallucination behaviors.
- `2-c-upload_audio.py`: Upload your finalized audio files to Google Cloud Storage.

#### 2. Model Inference
Submits asynchronous batch jobs to Vertex AI.
- `3-a-batch-inference.py`: Run batch inference on the primary evaluation dataset.
- `3-b-silence-batch-inference.py`: Run inference specifically on the silence-injected dataset.
- `99-cancel-batch-job.py`: Utility script to cancel running batch jobs if needed.

#### 3. Evaluation & Optimization
These scripts form the core of the automated prompt engineering loop:
- `4-evaluate-performance.py`: Compares ground truth to generated transcripts and calculates metrics (WER, CER, etc.).
- `5-critic.py`: Analyzes common failure modes using an LLM-based critic.
- `6-few-shot-optimize.py`: Automatically iterates on and generates new prompts by incorporating learned failure cases (few-shot prompting).

#### 4. Reporting & Deployment
- `7_generate_report.py`: Aggregates the evaluation results and generates an interactive `evaluation_report.html`.
- `8-firebase-deploy.py`: Bundles the HTML report and audio assets into the `public/` directory for deployment via Firebase Hosting.

## 📝 Folder Structure
```
.
├── src/                          # Core pipeline execution scripts
├── data/                         # Local storage for audio and transcription data
├── prompt/                       # Current and generated optimized prompts
├── execution_log/                # Logs containing Vertex AI Batch job details
├── batch_inference_results/      # Output inferences downloaded from Vertex AI
├── archive/                      # Archived experiments and Jupyter notebooks
├── run_pipeline.py               # Main pipeline execution entry point
├── .env                          # Environment variables configuration
└── README.md                     # You are here
```

## 🔒 Security Note
Do not commit `.env` or Firebase configuration files (`.firebaserc`, `firebase.json`) to version control if they contain sensitive keys or project IDs. These are already included in the `.gitignore` by default.
