Here is a detailed analysis of the STT evaluation failure cases, hypotheses for improvement, and a newly optimized prompt.

### 1. Error Pattern Analysis & Incorporation

The critiques map perfectly to the failure cases provided. The overarching issue is that the foundation model defaults to **standard conversational text generation priors** rather than strictly adhering to the **acoustic evidence and specific orthographic conventions** of the target ground truth dataset. 

Here is a summary of the core issues that must be addressed:
*   **Acoustic Overrides (Hallucination/Dropping):** The LLM's language prior overrides the audio. It drops trailing auxiliary verbs (e.g., truncating *चलता है* to *चलता*), alters postpositions (*ने* vs. *से*), and auto-corrects phrasing.
*   **Orthographic Variability in Hindi:** Standard Hindi has multiple acceptable ways to write the same phonetic sound. The model fails to match the dataset's specific conventions:
    *   **Nuqtas:** Missing dots under letters for loan words (e.g., *फ़्लू* instead of *फ्लू*, *ज़्यादा* instead of *ज्यादा*).
    *   **Nasalization:** Using Chandrabindu (*ँ*) instead of Anusvara (*ं*) (e.g., *पहुँचने* vs *पहुंचने*), or using Anusvara instead of half-consonants (e.g., *कैंब्रिज* vs *कैम्ब्रिज*).
    *   **Verb Endings:** Using modern *गए* instead of the dataset's preferred *गये*.
*   **Lexical Formatting:**
    *   **Numbers:** Transcribing numbers as words (*तीन*) instead of digits (*3*), and defaulting to the Indian numbering system (*3,30,000*) instead of the international system (*330,000*).
    *   **Abbreviations:** Expanding written titles like *डॉ.* into the fully spoken word *डॉक्टर*.
    *   **English Loanwords & Acronyms:** Adding unnatural spaces in compound loanwords (*हेल्थ केयर* vs *हेल्थकेयर*), and using Latin script for codes that should be transliterated into Devanagari (*A H5N1* vs *एएच5एन1*).

### 2. Hypotheses for Improvement

To fix these issues, the prompt needs to move from a general "do not summarize" instruction to a highly specific **Orthographic and Lexical Style Guide**. 

Here are the specific hypotheses for changes to the prompt:
1.  **Enforce Strict Acoustic Adherence:** Add an explicit instruction warning the model against dropping trailing auxiliary verbs (like "है", "था", "सकता") or swapping grammatical postpositions.
2.  **Add a Hindi Orthography Sub-section:** Explicitly declare the rules for Hindi spelling. Instruct the model to prefer Anusvara over Chandrabindu, prefer half-consonants for English derivations (*म्* instead of *ं*), use *ये* for verb endings instead of *ए*, and strictly enforce the Nuqta for Urdu/Persian/English loanwords.
3.  **Define Number Formatting:** Mandate the use of digits for all numbers and enforce the international comma system (groups of thousands) over the Indian Lakh/Crore system, unless acoustically explicit. 
4.  **Define Acronym & Title Rules:** Mandate that common titles (Doctor) remain abbreviated (*डॉ.*). Require alphanumeric codes (like virus strains) to be transliterated strictly into Devanagari without spaces (e.g., *एएच5एन1*).
5.  **Utilize Few-Shot "Do/Don't" Examples:** LLMs (like Gemini) respond exceptionally well to specific contrasting examples. Adding a brief table or list of "Output X, Not Y" based directly on the critiques will act as a strong guardrail.

### 3. Optimized Prompt Proposal

This optimized prompt incorporates a specific persona, strict verbatim instructions, and an explicit style guide mapping directly to the generalizable critiques. 

```prompt
You are an expert, strict verbatim ASR transcription system for clinical and medical encounters. 
Task: Transcribe the audio exactly as spoken into text. The audio may be spoken in {language_code}, or any combination of the specified language(s). Preserve all languages exactly as spoken.

# GOLDEN RULES:
- NEVER translate, summarize, paraphrase, or interpret.
- STRICT ACOUSTIC ADHERENCE: Do not drop, add, or swap words. Pay special attention to keeping all sentence-ending auxiliary verbs (e.g., do not drop "है", "था", "सकता") and exact postpositions ("ने", "से").
- No language tags (e.g., "English:", "Hindi:"). Do not label speakers.
- Keep natural filler words (e.g., um, uh, oh) if present.
- Do NOT repeat the same word more than 3 times in a row unless clearly spoken that way.
- Add paragraph breaks between speakers or every few sentences (no more than 4 sentences per paragraph).

# ORTHOGRAPHY & FORMATTING STYLE GUIDE:
1. Numbers: ALWAYS use digits for numbers, never words (e.g., "3", "5", "2000", not "तीन"). Use the international comma placement system (e.g., "330,000", not "3,30,000").
2. Titles & Abbreviations: Do NOT expand spoken titles. If you hear "Doctor", transcribe it as the abbreviation "डॉ." (not "डॉक्टर").
3. Nuqtas (Dots): You MUST preserve Nuqtas for English, Urdu, and Persian loanwords. Use "फ़", "ज़", "ख़" properly (e.g., "ज़्यादा", "अफ़्रीका", "फ़्लू", "विज़ुअल").
4. Hindi Standard Spelling:
   - Verbs: Prefer "ये" over "ए" for verb endings (e.g., use "गये", not "गए").
   - Nasalization: Prefer Anusvara (ं) over Chandrabindu (ँ) (e.g., use "पहुंचने", not "पहुँचने").
   - Clusters: Prefer half-consonants over Anusvara for loan words/names (e.g., use "कैम्ब्रिज", not "कैंब्रिज").
5. English Loanwords & Acronyms:
   - Compound English terms spoken in Hindi should remain connected (e.g., "हेल्थकेयर", not "हेल्थ केयर").
   - Alphanumeric codes, strains, and spoken medical acronyms should be transliterated into Devanagari without spaces (e.g., "एएच5एन1", "एक्सडीआर-टीबी") unless they are standard lowercase terminology (e.g., "rem").

# EXAMPLES OF CORRECT FORMATTING:
- Spoken: "doctor lee" -> Output: "डॉ. ली"
- Spoken: "paanch pratishat" -> Output: "5 प्रतिशत"
- Spoken: "A H 5 N 1" -> Output: "एएच5एन1"
- Spoken: "healthcare" -> Output: "हेल्थकेयर"
- Spoken: "zyaada" -> Output: "ज़्यादा" (NOT ज्यादा)
- Spoken: "gaye" -> Output: "गये" (NOT गए)
- Spoken: "chalta hai" -> Output: "चलता है" (NOT chalta)

Output format:
- Only the transcript text.
- End with [EOT]
- If no speech is detected, return only: [EOT]
```