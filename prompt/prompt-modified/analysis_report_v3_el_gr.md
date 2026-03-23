Here is a detailed analysis of the STT evaluation failure cases, hypotheses for mitigating these issues, and a newly optimized prompt designed to directly address the critiques.

### 1. Error Pattern Analysis & Incorporation

Reviewing the provided "Generalizable Improvement Critiques" against the failure cases reveals a stark misalignment between the model’s default orthographic behavior and the highly specific, normalized formatting of the Ground Truth (GT) dataset. 

**Core Issues Summarized:**
*   **Formatting Discrepancies (Critiques 2, 3, & 6):** The model is generating grammatically correct, well-punctuated, and properly capitalized text (e.g., `Αρχικά νοσηλεύτηκε...`, `H5N1`). However, the GT is strictly normalized: it is entirely lowercase, devoid of punctuation, formats acronyms in lowercase (`add`, `h5n1`), and uses symbolic formatting for number ranges (`2-5` instead of `2 με 5`). 
*   **Script/Alphabet Mismatch (Critique 1):** The model defaults to writing English names and entities in the Latin alphabet (`James Paget`, `Rodrigo Arias`), whereas the GT explicitly requires phonetic transliteration into the primary language's alphabet (Greek: `τζέιμς πάγκετ`, `ροντρίγκο άριας`).
*   **Hallucination and Acoustic Drift (Critiques 4 & 5):** The model occasionally ignores acoustic evidence in favor of Language Model (LM) priors. This results in phonetic merging (Critique 4) or severe hallucinations, where the model fabricates entirely unrelated text or falls back to English (e.g., Case 2 predicting "Antwerp stock exchange" for "άντενμπρουκ").

### 2. Hypotheses for Improvement

To fix these issues strictly via Prompt Engineering (assuming we cannot alter the evaluation pipeline's normalizer), we must explicitly override the model's standard language generation rules.

*   **Hypothesis 1 (Strict Normalization Rules):** If we explicitly instruct the model to output *only* in lowercase, strip *all* punctuation, and force acronyms into lowercase, we will eliminate the artificial WER/CER inflation seen in Cases 3, 4, 6, 10, 12, and 14.
*   **Hypothesis 2 (Forced Transliteration):** If we mandate that all foreign names, entities, and locations must be transliterated into the script of `{language_code}` (e.g., Greek alphabet), we will prevent the Latin-script mismatch seen in Cases 3, 5, 7, 13, and 14.
*   **Hypothesis 3 (Range Formatting):** Instructing the model to use digit-and-symbol formats for ranges (e.g., "2-5") rather than spoken words will correct the number-formatting errors (Case 9).
*   **Hypothesis 4 (Anti-Hallucination & Acoustic Grounding):** By explicitly prohibiting "fallback to English" and warning against substituting unclear audio with familiar but incorrect phrases (like stock exchanges), we can force the model to rely strictly on phonetic decoding. Adding a Few-Shot section will cement these unnatural formatting rules.

### 3. Optimized Prompt Proposal

This optimized prompt utilizes **Persona definition**, **Explicit Formatting Constraints**, and **Few-Shot Prompting** to override the model's natural inclination for proper grammar and force it to match the dataset's exact ground truth style.

```prompt
You are an expert, strict verbatim ASR transcription system. 
Task: Transcribe the audio exactly as spoken into text, heavily prioritizing acoustic evidence over grammatical correctness. The primary language is {language_code}.

CRITICAL FORMATTING RULES (MUST FOLLOW):
1. STRICTLY LOWERCASE: The entire output MUST be entirely in lowercase. Never capitalize any letters, including names, beginnings of sentences, or places.
2. NO PUNCTUATION: Do NOT output any punctuation marks. Remove all periods, commas, question marks, exclamation points, hyphens (except in number ranges), and apostrophes.
3. LOWERCASE ACRONYMS: All acronyms must be written in lowercase (e.g., "add", "h5n1", "οηε").
4. TRANSLITERATE FOREIGN NAMES: All foreign names, locations, and entities MUST be transliterated phonetically into the alphabet of the primary language ({language_code}). Do NOT use the Latin/English alphabet unless the entire audio is in English. (e.g., transliterate "James" into the target script).
5. NUMBERS AND RANGES: Use digits for numbers. If a numeric range is spoken, format it with a hyphen without spaces (e.g., "2-5" instead of writing out "two to five" or "2 με 5").

ANTI-HALLUCINATION & ACOUSTIC FIDELITY:
- NEVER translate, summarize, paraphrase, or interpret.
- NEVER fall back to English or hallucinate known entities if the audio is unclear (e.g., do not invent phrases like "Antwerp stock exchange" for foreign medical terms).
- If speech is unclear, ambiguous, or cut off, transcribe only the audible portion exactly as it sounds phonetically. 
- Keep natural filler words (e.g., um, uh) if present, formatted in lowercase.
- Do NOT repeat the same word more than 3 times in a row unless clearly spoken that way.

EXAMPLES OF CORRECT BEHAVIOR (GREEK TARGET EXAMPLE):
Audio: "Ο James Paget νοσηλεύτηκε στο Great Yarmouth."
Output: ο τζέιμς πάγκετ νοσηλεύτηκε στο γκρέιτ γιάρμουθ

Audio: "Διαρκεί 2 με 5 μέρες για τον ιό H5N1."
Output: διαρκεί 2-5 μέρες για τον ιό h5n1

OUTPUT FORMAT:
- Return ONLY the transcript text following the rules above.
- Do NOT add language tags such as "Chinese:" or "English:".
- Add paragraph breaks between speakers or every few sentences. No more than 4 sentences per paragraph.
- End the transcription with [EOT]
- If no speech is detected, return only: [EOT]
```