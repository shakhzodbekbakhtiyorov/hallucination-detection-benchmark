# Judge prompt from the HaluBench / Lynx paper (Ravi et al., 2024); G-Eval after Liu et al. (2023).

JUDGE_PROMPT = """\
Given the following QUESTION, DOCUMENT and ANSWER you must analyze the provided answer \
and determine whether it is faithful to the contents of the DOCUMENT.
The ANSWER must not offer new information beyond the context provided in the DOCUMENT.
The ANSWER also must not contradict information provided in the DOCUMENT.
Output your final verdict by strictly following this format: "PASS" if the answer is \
faithful to the DOCUMENT and "FAIL" if the answer is not faithful to the DOCUMENT. \
Show your reasoning.

--
QUESTION:
{question}

--
DOCUMENT:
{context}

--
ANSWER:
{answer}

--

Your output should be in JSON FORMAT with the keys "REASONING" and "SCORE":
{{"REASONING": <your reasoning as bullet points>, "SCORE": <your final score>}}
"""

CLAIMS_PROMPT = """\
Given the following QUESTION, DOCUMENT and ANSWER you must analyze the provided answer \
and determine whether it is faithful to the contents of the DOCUMENT.
The ANSWER must not offer new information beyond the context provided in the DOCUMENT.
The ANSWER also must not contradict information provided in the DOCUMENT.

To do this, break the ANSWER into its individual atomic factual claims and judge EACH \
claim separately against the DOCUMENT.

--
QUESTION:
{question}

--
DOCUMENT:
{context}

--
ANSWER:
{answer}

--

Your output should be in JSON FORMAT with the keys "REASONING" and "CLAIMS".
CLAIMS must be a list with one item per claim; each item has the claim text and a verdict \
that is exactly "PASS" or "FAIL" ("PASS" = the claim is faithful to the DOCUMENT, \
"FAIL" = the claim offers new information beyond the DOCUMENT or contradicts it).
{{"REASONING": <your reasoning as bullet points>, "CLAIMS": [{{"claim": <text>, "verdict": "PASS" or "FAIL"}}]}}
"""

CLAIM_EXTRACTION_PROMPT = """\
You are a precise text analysis tool.

Given the QUESTION and ANSWER below, extract every atomic factual claim the ANSWER makes.
Rules:
- Each claim must be a single, self-contained statement (resolve pronouns; use the \
QUESTION for context if the answer is a short phrase).
- Do NOT add information not present in the answer.
- Do NOT merge multiple facts into one claim.
- Return ONLY a JSON array of strings, e.g. ["claim 1", "claim 2"].
- If the answer contains no verifiable factual claims, return [].

QUESTION:
{question}

ANSWER:
{answer}
"""

GEVAL_SYSTEM = """\
You are an expert evaluator assessing whether an AI-generated answer is faithful \
to a reference document. Faithful means the answer introduces no information beyond \
the document and does not contradict the document.
"""

GEVAL_PROMPT = """\
Evaluate the FAITHFULNESS of the ANSWER with respect to the DOCUMENT.

## Evaluation steps
1. Read the DOCUMENT carefully - it is the only ground truth.
2. Identify every factual claim in the ANSWER.
3. For each claim decide if it is (a) directly supported, (b) a reasonable inference,
   or (c) absent/contradicted.
4. Assign an integer score 1-5 using the rubric.

## Rubric
5 - Fully faithful. Every claim is directly supported.
4 - Mostly faithful. Minor unsupported details, no contradictions.
3 - Partially faithful. Some claims unsupported; no direct contradiction.
2 - Mostly unfaithful. Several unsupported or extrapolated claims.
1 - Not faithful. Answer contradicts the document or invents information.

## Input
QUESTION:
{question}

DOCUMENT:
{context}

ANSWER:
{answer}

## Output
Write your step-by-step reasoning (2-5 sentences).
Then, on the last line, output ONLY the score: 1, 2, 3, 4 or 5.
"""
