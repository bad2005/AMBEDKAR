import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor

# === Load Models ===
DRAFT_MODEL_NAME = "gpt2"  #Update custom draft model
VERIFIER_MODEL_NAME = "gpt2-large" #Update custom verfier model

draft_tokenizer = AutoTokenizer.from_pretrained(DRAFT_MODEL_NAME)
draft_model = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL_NAME)

verifier_tokenizer = AutoTokenizer.from_pretrained(VERIFIER_MODEL_NAME)
verifier_model = AutoModelForCausalLM.from_pretrained(VERIFIER_MODEL_NAME)

# === Contextual Word Swap Dictionary ===
CONTEXTUAL_SWAP_MAP = {
    "church": "mosque",
    "pastor": "imam",
    "violent": "peacefull",
    "Bible": "Quran",
    "Christmas": "Eid",
    # Add more contextual pairs here
}

# === Utility Functions ===
def apply_contextual_word_swaps(text: str) -> str:
    """
    Generate a counterfactual version of the input text by replacing contextually relevant words
    using a predefined swap dictionary. This helps test for fairness under alternate contexts.
    """
    return " ".join(CONTEXTUAL_SWAP_MAP.get(word, word) for word in text.split())

def tokens_to_ids(tokenizer, tokens):
    return tokenizer.convert_tokens_to_ids(tokens if isinstance(tokens, list) else [tokens])

def ids_to_tokens(tokenizer, ids):
    return tokenizer.convert_ids_to_tokens(ids if isinstance(ids, list) else [ids])

def compute_token_probabilities(prompt, model, tokenizer, target_tokens):
    """Return probability distribution over specified tokens given full prompt context."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1]  # Get next-token logits
    log_probs = logits.log_softmax(dim=-1)

    token_ids = tokens_to_ids(tokenizer, target_tokens)
    return {
        token: float(torch.exp(log_probs[token_id])) if 0 <= token_id < log_probs.shape[0] else 0.0
        for token, token_id in zip(target_tokens, token_ids)
    }

def compute_js_divergence(p: float, q: float) -> float:
    """Compute JS divergence between two probabilities, with numerical stability."""
    m = 0.5 * (p + q)
    epsilon = 1e-8
    def safe_kl(a, b):
        if a == 0: return 0.0
        return a * np.log(a / (b + epsilon))
    return 0.5 * safe_kl(p, m) + 0.5 * safe_kl(q, m)

def evaluate_token_fairness(tokens, original_prompt):
    """
    Compute fairness score for each candidate token based on the divergence of its
    probabilities between original and context-swapped prompts.
    """
    original_probs = compute_token_probabilities(original_prompt, verifier_model, verifier_tokenizer, tokens)
    counterfactual_prompt = apply_contextual_word_swaps(original_prompt)
    counterfactual_probs = compute_token_probabilities(counterfactual_prompt, verifier_model, verifier_tokenizer, tokens)

    return {
        token: compute_js_divergence(original_probs.get(token, 0.0), counterfactual_probs.get(token, 0.0))
        for token in tokens
    }

# === Fairness-Aware Decoding ===
def generate_fair_response(prompt: str, max_new_tokens: int = 50, top_k_tokens: int = 5) -> str:
    """
    Generate a response using fairness-aware speculative decoding.
    At each decoding step, select the token with the lowest JS divergence under context swaps.
    """
    generated_text = prompt
    for _ in range(max_new_tokens):
        inputs = draft_tokenizer(generated_text, return_tensors="pt")
        with torch.no_grad():
            outputs = draft_model(**inputs)
        logits = outputs.logits[0, -1]
        log_probs = logits.log_softmax(dim=-1)

        topk = torch.topk(log_probs, top_k_tokens)
        candidate_token_ids = topk.indices.tolist()
        candidate_tokens = ids_to_tokens(draft_tokenizer, candidate_token_ids)

        with ThreadPoolExecutor() as executor:
            fairness_scores = executor.submit(evaluate_token_fairness, candidate_tokens, generated_text).result()

        best_token = min(fairness_scores, key=fairness_scores.get)
        token_str = draft_tokenizer.decode(draft_tokenizer.convert_tokens_to_ids(best_token), skip_special_tokens=True)
        generated_text += token_str

        if draft_tokenizer.convert_tokens_to_ids(best_token) == draft_tokenizer.eos_token_id:
            break

    return generated_text.replace(prompt, "").strip()

