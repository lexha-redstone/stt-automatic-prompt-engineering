Here is a detailed analysis of the STT evaluation failure cases, hypotheses for improvement, and a newly optimized prompt designed to mitigate these exact issues.

### 1. Error Pattern Analysis & Incorporation

**Alignment of Critiques with Failure Cases:**
The provided critiques directly address the core failure modes seen in the Ground Truth vs. Prediction examples:
*   **Critique 1 (Phonetic/Homophone Confusion):** Highly visible in Case 4 (where the model hears "האיטי" [Haiti] but outputs "היא תלונה תקלה" [gibberish/phonetic misfire]) and Case 11 ("לנטיין מומחה" becomes "לנקרי, מנהל החנות"). The acoustic model dominates the language model, leading to phonetically plausible but contextually absurd outputs.
*   **Critique 2 (Foreign Words/Acronyms & BIDI):** Evident in Case 2, where "HIV" is attached to the wrong word, and Case 3, where "xdr-tb" is thrown to the very end of the sentence. Case 9 shows "ZMapp" phonetically butchered into "זימבפ" instead of preserved. Mixing Right-to-Left (RTL) Hebrew with Left-to-Right (LTR) English causes the model's token alignment to break.
*   **Critique 3 (Paraphrasing & Perspective Shifts):** Distinctly seen in Case 7 ("הוא לא היה יכול" [he could not] shifts to "אני רוצה" [I want]) and Case 15 ("האיבר הראשי" [main organ] becomes "הדבר הכי חשוב" [the most important thing]). The model slips into a conversational, generative "assistant" persona rather than acting as a strict transcription tool.
*   **Critique 4 (Hallucination & Reordering):** Shown in Case 9, where the model completely hallucinates the word "פלסבו" (placebo) likely because the context was a medical trial, relying on its internal knowledge rather than the audio track. Sentences are also badly scrambled chronologically.
*   **Critique 5 (Numbers & Units):** While explicit cases of pound/kg conversions weren't in the top 15 list, the underlying issue stems from the generative model attempting to "be helpful" by standardizing or localizing data rather than simply listening.

**Core Issues Summary:**
The current prompt defines the *task* but does not constrain the *generative behavior* of the foundation model. Gemini-Flash is acting like a helpful assistant analyzing audio, rather than a rigid, exact transcriber. It relies too heavily on raw acoustics without semantic checking (Critique 1), fails at formatting mixed-direction text (Critique 2), rewrites text (Critiques 3 & 4), and normalizes data (Critique 5).

---

### 2. Hypotheses for Improvement

To fix these issues, the prompt must explicitly target the generative language model (LLM) tendencies of the Gemini foundation model. Here is how we will change the prompt:

1.  **Address Paraphrasing & Perspective (Critique 3):** We will add explicit directives: *"Do NOT alter pronouns (e.g., changing 'he' to 'I'), subjects, or verb tenses."*
2.  **Address Semantic Coherence (Critique 1):** We will instruct the model to use the surrounding context to validate phonetic guesses: *"Use semantic context to disambiguate homophones and phonetically similar words. The output must make logical and grammatical sense."*
3.  **Address BIDI & Foreign Words (Critique 2):** We must add specific layout and character handling rules: *"Ensure correct Right-to-Left (RTL) and Left-to-Right (LTR) token ordering (BIDI text alignment) so English words appear in the exact chronological position spoken. Preserve English acronyms in their original Latin characters."*
4.  **Address Temporal Reordering & Hallucinations (Critique 4):** We will add a strict chronology constraint: *"Strictly follow the chronological order of the audio. Do NOT hallucinate, guess, or add words based on context."*
5.  **Address Unit Conversions (Critique 5):** We will add a mathematical and measurement constraint: *"NEVER convert measurements (e.g., pounds to kg) or perform mathematical operations. Transcribe numbers exactly as spoken."*

---

### 3. Optimized Prompt Proposal

Here is the proposed prompt. It uses a structured format with clear headings, explicit "DO NOT" constraints, and incorporates targeted guidelines to address the generalizable improvements.

```prompt
You are an expert STT (Speech-to-Text) transcription engine performing strict verbatim transcription for a clinical encounter. 

Task: Transcribe the audio exactly as spoken into text. The audio may be spoken in {language_code}, or any combination of the specified language(s). 

CORE DIRECTIVES:
- STRICT VERBATIM: Output exactly what is spoken. You must NEVER translate, summarize, paraphrase, or interpret.
- NO HALLUCINATIONS: Do NOT hallucinate, guess, or add words (e.g., adding medical terms like "placebo" that were not spoken).
- CHRONOLOGICAL ORDER: Strictly follow the temporal sequence of the audio. Do not rearrange sentence structures or move words to the end of the sentence.
- NO PERSPECTIVE SHIFTS: Do NOT alter pronouns (e.g., changing "he could not" to "I want"), subjects, or verb tenses. Transcribe from the exact perspective of the speaker.

LANGUAGE, BIDI, & VOCABULARY RULES:
- SEMANTIC COHERENCE: Use semantic context to disambiguate homophones and phonetically similar words. Avoid substituting contextually incorrect words just because they sound similar (e.g., do not transcribe "vision" as "wife" if it makes no logical sense).
- MIXED LANGUAGES & BIDI: If multiple languages are spoken, keep them exactly as spoken. Ensure correct Right-to-Left (RTL) and Left-to-Right (LTR) token ordering (BIDI text alignment) so English or foreign words appear in the exact chronological position they were spoken.
- ACRONYMS & NAMES: Preserve English acronyms (e.g., HIV, MRI, ZMapp, xdr-tb) in their original Latin characters. Transcribe foreign names phonetically exactly as they sound without breaking them into unrelated words.
- NO TAGS: Do NOT group, label, or add language tags such as "Hebrew:" or "English:". Do NOT add explanations or commentary.

NUMBERS & FORMATTING:
- UNITS & NUMBERS: Transcribe numbers and units exactly as spoken. NEVER convert measurements (e.g., do not change pounds to kilograms) and do not perform mathematical operations.
- GRAMMAR & FILLERS: Do NOT clean up grammar. Keep natural filler words (e.g., um, uh, oh) if present. 
- REPETITION: Do NOT repeat the same word more than 3 times in a row unless clearly spoken that way.
- UNCLEAR SPEECH: If speech is unclear or cut off, transcribe only the audible portion. Do not attempt to complete the sentence.
- PARAGRAPHING: Add paragraph breaks between speakers or every few sentences. No more than 4 sentences per paragraph.

OUTPUT FORMAT:
- Output ONLY the transcript text.
- End the transcription with exactly: [EOT]
- If no speech is detected, return only: [EOT]
```