Here is a detailed analysis of the STT evaluation failure cases, hypotheses for improvement, and a newly optimized prompt designed to mitigate the identified errors.

### 1. Error Pattern Analysis & Incorporation

The provided "Generalizable Improvement Critiques" map perfectly to the underlying causes of the high Word Error Rate (WER) and Character Error Rate (CER) in the 15 failure cases. The core issues stem from a misalignment between the model’s natural transcription tendencies (grammatically correct, phonetically literal, and fully punctuated) and the specific idiosyncrasies of the ground truth dataset. 

Here is how the critiques align with the failure cases:
*   **Phonetic vs. Semantic Transcription (Critique 1):** In Cases 5, 7, 10, 13, and 14, the model acts as a "dumb ear," transcribing exactly what it hears phonetically (e.g., "궁극주의" instead of "군국주의", "의과 애들은" instead of "외과의들은"). Without instructions to weigh the semantic context of a clinical/historical encounter, it fails to disambiguate homophones or slurred speech.
*   **Spacing Rules / 띄어쓰기 (Critique 2):** In Korean, compound nouns can often be spaced or combined. The model defaults to spacing them (e.g., "모세 혈관", "응급 처치"), whereas the ground truth dataset heavily prefers combining them ("모세혈관", "응급처치"). This creates artificial WER inflation.
*   **Number & Range Formatting (Critique 3):** Cases 3, 4, and 12 highlight a classic STT formatting mismatch. The model transcribes numbers phonetically ("두 명", "33만 명", "10분에서 60분"), while the ground truth uses strict Arabic numeral and symbolic formatting ("2명", "330,000명", "10-60분"). 
*   **Acronyms & Dataset Artifacts (Critique 4):** Cases 1, 2, 8, and 12 reveal inconsistencies in how foreign words and acronyms are handled. The dataset contains weird artifacts (like unpronounced concatenated English text in Case 1: "야머스great yarmouth") and lowercase acronyms. Establishing a standard rule for the model will prevent erratic guessing.
*   **Punctuation (Critique 5):** This is the largest driver of WER/CER across almost *all* cases. The model naturally applies standard punctuation (periods, commas), but the ground truth dataset is entirely unpunctuated. Every period generated counts as an insertion error.

### 2. Hypotheses for Improvement

To fix these issues, the prompt must shift from a generic "transcribe exactly as spoken" instruction to a highly constrained, formatting-aware set of rules tailored to this specific dataset's quirks.

*   **Hypothesis 1 (Contextual Disambiguation):** By explicitly instructing the model to act as a semantic checker, it will use the surrounding sentence context to select the logically correct word among Korean homophones (e.g., inferring "외과의들은" [surgeons] instead of "의과 애들은" [medical kids] in a clinical context).
*   **Hypothesis 2 (Zero Punctuation):** Adding an absolute negative constraint (`Do NOT generate any punctuation marks`) will immediately eliminate the systemic insertion errors inflating the WER across all cases.
*   **Hypothesis 3 (Strict Normalization for Numbers/Ranges):** By dictating the use of Arabic numerals (`2`, `330,000`) and hyphens for ranges (`10-60`), the model will bypass phonetic transcription for numerical values and output the math-based format the dataset expects.
*   **Hypothesis 4 (Compound Noun Spacing Bias):** Directing the model to collapse compound nouns into single words without spaces will align its spacing algorithm with the ground truth's grammatical preferences.
*   **Hypothesis 5 (Foreign Word Standardization):** Setting a strict policy (Acronyms = English Uppercase, Spoken foreign terms = Hangul, Ignore unpronounced artifacts) gives the LLM a deterministic path when encountering mixed-language entities.

### 3. Optimized Prompt Proposal

I have integrated the critiques into a robust prompt. I utilized **Negative Constraints** (capitalized "NEVER/NO") for strict boundaries, and a **Few-Shot / Pattern-Matching** section to demonstrate exactly how numbers, spacing, and punctuation should be handled.

```prompt
You are an expert ASR transcription system performing strict verbatim transcription for a clinical encounter. 
Task: Transcribe the audio exactly as spoken into text. The audio may be spoken in {language_code}, or any combination of the specified language(s). 

CRITICAL FORMATTING AND NORMALIZATION RULES:
- Semantic Coherence Check: Use surrounding sentence context to disambiguate phonetically similar Korean words and homophones. Ensure the final transcription is logically and semantically sound (e.g., transcribe "외과의들은" instead of "의과 애들은", "군국주의" instead of "궁극주의", "뇌 병리" instead of "네, 병리").
- NO PUNCTUATION: Do NOT generate ANY punctuation marks whatsoever. Do not use periods, commas, quotation marks, question marks, or exclamation points. Output the text in a completely unpunctuated style.
- Number Formatting: ALWAYS transcribe numbers using Arabic numerals (e.g., 2, 330000) rather than spelling them out in Hangul. Use hyphens to denote ranges (e.g., 10-60) instead of spelling out range indicators like "에서".
- Spacing (띄어쓰기) Rules: Adhere to standard Korean spacing rules, but strictly prefer grouping compound nouns together without spaces (e.g., "평화유지군", "응급처치", "모세혈관", "신경독소") and attach auxiliary verbs to main verbs.
- Foreign Words & Acronyms: Transcribe acronyms in uppercase English (e.g., UN, XDR-TB). If a foreign word is spoken, transcribe it phonetically in Hangul unless specified otherwise. Ignore and do NOT generate unpronounced English spelling artifacts (e.g., do not output "야머스great yarmouth", output "야머스" if only the Korean was spoken).

GENERAL RULES:
- You must NEVER translate, summarize, paraphrase, or interpret.
- Preserve the original spoken language exactly as it appears, including mixed languages. Do NOT add language tags (e.g., "Chinese:" or "English:").
- Do NOT add explanations, commentary, or clean up grammar.
- Keep natural filler words (e.g., um, uh, oh) if present.
- Do NOT repeat the same word more than 3 times in a row unless clearly spoken that way.
- If speech is unclear or cut off, transcribe only the audible portion.
- Add paragraph breaks between speakers or every few sentences. No more than 4 sentences per paragraph.

Output format:
- Only the transcript text.
- End with [EOT]
- If no speech is detected, return only: [EOT]
```