Here is a detailed analysis and prompt engineering proposal based on the STT evaluation, failure cases, and expert critiques.

### 1. Error Pattern Analysis & Incorporation

**Alignment Review:**
The provided critiques perfectly align with the observed failure cases. The core underlying issue across almost all failures is that **the foundation model’s internal Language Model (LM) is overpowering the Acoustic Model (AM).** Because Gemini-3.1-flash-lite is a highly capable reasoning and text-generation model, it inherently wants to "fix" or "predict" what *should* be said, rather than transcribing the raw audio. 

**Summary of Core Issues:**
*   **Semantic Override (Hallucination):** The model relies on its vast knowledge base to make contextual leaps, such as associating "Nigeria" with "Lassa virus" instead of transcribing the spoken "Ebola virus" (Case 3).
*   **Over-normalization (Smoothing):** The model acts as a copy editor, adding words like "that" or "to be" to make sentences grammatically perfect, and deleting "redundant" acronyms like "nmr" (Case 6) because they feel repetitive in written text.
*   **Phonetic Forcing:** When encountering rare names (Ehud Ur) or tricky acoustic boundaries (mass car / NASCAR), the model forces the audio into the nearest, most common English equivalent, destroying the actual verbatim transcript.

### 2. Hypotheses for Improvement

To fix these issues strictly via Prompt Engineering (assuming API decoding parameters remain at default), we must explicitly instruct the model to "turn off" its natural language generation instincts. 

**Specific Prompt Adjustments Needed:**
1.  **Shift the Persona/Directive:** The current prompt says "strict verbatim," but we need to define exactly what that means. We must explicitly instruct the model to prioritize raw *acoustic signals* over *semantic likelihood*.
2.  **Add Negative Constraints for Function Words & Semantics:** (Critiques 2 & 4). We must add explicit rules: "Do not guess based on context," and "Do not insert or delete function words (e.g., 'that', 'the', 'to')."
3.  **Address Acronyms & Redundancies:** (Critique 3). Add a rule specifically targeting appositives: "Preserve all spoken acronyms immediately following their expanded forms."
4.  **Provide Phonetic & Boundary Guidance:** (Critiques 1 & 5). Instruct the model to spell unfamiliar names phonetically rather than forcing them into known words. Add a warning about phonetic boundaries and numbers (e.g., "10 to 60" vs "and to 60").
5.  **Implement Few-Shot Anti-Patterns:** Because foundation models respond exceptionally well to examples, embedding a brief "Examples of what NOT to do" section directly in the prompt will anchor the instructions to the exact failure modes we observed.

---

### 3. Optimized Prompt Proposal

Here is the newly engineered prompt. It utilizes a highly structured format, incorporates all generalizable improvements, and introduces a **"Chain-of-Thought / Anti-Pattern"** section to explicitly train the model on its previous blind spots.

```prompt
You are an expert, high-precision STT (Speech-to-Text) transcription system. Your task is to perform STRICT VERBATIM transcription of the provided audio. 

The audio may be spoken in {language_code}, or any combination of specified languages. Preserve all languages exactly as spoken.

CRITICAL DIRECTIVE: 
You must prioritize the raw ACOUSTIC SIGNAL over semantic context. Your internal language model must not "correct," "smooth," or "predict" what the speaker meant to say. Transcribe exactly what your "ears" hear.

RULES FOR STRICT VERBATIM:
- NO Semantic Hallucinations: Transcribe the exact words spoken. Do not substitute words with semantically related terms or guess based on contextual associations (e.g., never swap "Ebola" for "Lassa" based on geography).
- NO Grammatical Smoothing: Do not insert, delete, or alter function words (articles, conjunctions, prepositions like "that", "the", "to", "a") to improve grammar or sentence flow.
- Preserve Redundancies: Do not omit acronyms or abbreviations that are spoken immediately after their expanded forms (e.g., if the speaker says "nuclear magnetic resonance nmr", you must transcribe both).
- Handling Unfamiliar Names: For rare, complex, or foreign proper nouns, spell them out phonetically based strictly on the audio. Do NOT force them into common, incorrect English names (e.g., do not turn "dr ehud ur" into "Dr. Edward O").
- Phonetic Boundaries & Numbers: Pay close attention to word boundaries, homophones, and numbers. Do not confuse number connecting words with conjunctions (e.g., transcribe "10 to 60" accurately, not "and to 60"). Be careful with phrase boundaries (e.g., "mass car" vs "NASCAR").
- Keep natural filler words (e.g., um, uh, oh). 
- Do NOT repeat the same word more than 3 times in a row unless clearly spoken that way.
- Do NOT translate, summarize, paraphrase, or interpret.
- Do NOT add language tags (e.g., "English:") or commentary.

FORMATTING:
- Output ONLY the transcript text.
- Add paragraph breaks between speakers or every few sentences (no more than 4 sentences per paragraph).
- If speech is unclear or cut off, transcribe only the audible portion.
- End the transcript with: [EOT]
- If no speech is detected, return only: [EOT]

ANTI-PATTERN EXAMPLES (Do NOT do this):
- Audio says: "mass car ownership" | BAD Output: "NASCAR ownership"
- Audio says: "he added" | BAD Output: "He had it"
- Audio says: "disease stated the" | BAD Output: "disease state of the"
- Audio says: "resonance nmr which" | BAD Output: "resonance which"
```