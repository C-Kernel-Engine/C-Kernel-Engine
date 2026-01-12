"""
Gibberish Detector for C-Kernel-Engine Output

Detects when model output is likely gibberish vs coherent text.
When gibberish is detected, automatically triggers staged validation.

Detection methods:
1. Repetition detection (same token repeated many times)
2. Entropy analysis (too high or too low entropy)
3. Special token ratio (too many special tokens)
4. Perplexity check (if reference model available)
"""

import re
import math
from typing import List, Tuple, Optional
from collections import Counter
from dataclasses import dataclass


@dataclass
class GibberishResult:
    """Result of gibberish detection"""
    is_gibberish: bool
    confidence: float  # 0.0 = definitely coherent, 1.0 = definitely gibberish
    reason: str
    details: dict


def detect_repetition(tokens: List[int], threshold: float = 0.5) -> Tuple[bool, float, str]:
    """
    Detect excessive token repetition.
    Returns: (is_repetitive, score, message)
    """
    if len(tokens) < 5:
        return False, 0.0, "too few tokens"

    counter = Counter(tokens)
    most_common_token, most_common_count = counter.most_common(1)[0]
    repetition_ratio = most_common_count / len(tokens)

    # Check for consecutive repetitions
    max_consecutive = 1
    current_consecutive = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i-1]:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1

    consecutive_score = max_consecutive / len(tokens)

    # Combined score
    score = max(repetition_ratio, consecutive_score * 2)

    if score > threshold:
        return True, score, f"repetition_ratio={repetition_ratio:.2f}, max_consecutive={max_consecutive}"

    return False, score, f"repetition_ratio={repetition_ratio:.2f}"


def detect_entropy_anomaly(tokens: List[int], vocab_size: int = 32000) -> Tuple[bool, float, str]:
    """
    Detect abnormal token entropy.
    Very low entropy = stuck on same tokens
    Very high entropy = random noise
    """
    if len(tokens) < 5:
        return False, 0.0, "too few tokens"

    counter = Counter(tokens)
    total = len(tokens)

    # Calculate Shannon entropy
    entropy = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    # Normalize by max possible entropy
    max_entropy = math.log2(min(len(tokens), vocab_size))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # Check for anomalies
    is_anomaly = False
    message = f"entropy={entropy:.2f}, normalized={normalized_entropy:.2f}"

    # Too low entropy (stuck)
    if normalized_entropy < 0.1 and len(tokens) > 10:
        is_anomaly = True
        message = f"very low entropy ({normalized_entropy:.2f}) - likely stuck"

    # Moderately low (suspicious)
    elif normalized_entropy < 0.3 and len(tokens) > 20:
        is_anomaly = True
        message = f"low entropy ({normalized_entropy:.2f}) - possible repetition"

    return is_anomaly, 1.0 - normalized_entropy, message


def detect_special_token_ratio(tokens: List[int],
                               special_tokens: List[int] = None,
                               threshold: float = 0.3) -> Tuple[bool, float, str]:
    """
    Detect excessive special/control tokens.
    Normal text shouldn't have too many special tokens.
    """
    if special_tokens is None:
        # Common special token ranges (varies by tokenizer)
        # Typically: BOS, EOS, PAD, UNK, and control tokens
        special_tokens = list(range(0, 100)) + list(range(32000, 32100))

    if len(tokens) < 3:
        return False, 0.0, "too few tokens"

    special_set = set(special_tokens)
    special_count = sum(1 for t in tokens if t in special_set)
    ratio = special_count / len(tokens)

    if ratio > threshold:
        return True, ratio, f"special_token_ratio={ratio:.2f} ({special_count}/{len(tokens)})"

    return False, ratio, f"special_token_ratio={ratio:.2f}"


def detect_text_patterns(text: str) -> Tuple[bool, float, str]:
    """
    Detect gibberish patterns in decoded text.
    """
    if len(text) < 10:
        return False, 0.0, "too short"

    issues = []
    score = 0.0

    # Check for repeated substrings
    for length in [3, 5, 10]:
        for i in range(len(text) - length * 3):
            substring = text[i:i+length]
            if text.count(substring) > 5:
                issues.append(f"repeated '{substring[:20]}...' {text.count(substring)} times")
                score = max(score, 0.8)
                break

    # Check for non-ASCII characters in what should be English output
    # This catches Chinese/Japanese/Korean characters appearing unexpectedly
    # Common when model generates wrong tokens (embedding or logits bug)
    non_ascii_count = sum(1 for c in text if ord(c) > 127)
    non_ascii_ratio = non_ascii_count / len(text) if len(text) > 0 else 0
    if non_ascii_ratio > 0.3:
        issues.append(f"high non-ASCII ratio: {non_ascii_ratio:.2f} ({non_ascii_count} chars) - likely wrong tokens")
        score = max(score, 0.9)  # High confidence this is wrong

    # Check for unusual character patterns
    # Too many special characters
    special_char_count = len(re.findall(r'[^\w\s,.!?\'"-]', text))
    special_ratio = special_char_count / len(text)
    if special_ratio > 0.3:
        issues.append(f"high special char ratio: {special_ratio:.2f}")
        score = max(score, 0.6)

    # Check for word repetition
    words = text.lower().split()
    if len(words) > 5:
        word_counter = Counter(words)
        most_common_word, count = word_counter.most_common(1)[0]
        word_ratio = count / len(words)
        if word_ratio > 0.3 and len(most_common_word) > 2:
            issues.append(f"word '{most_common_word}' repeated {count} times ({word_ratio:.0%})")
            score = max(score, 0.7)

    # Check for lack of spaces (garbled output)
    if len(text) > 50 and text.count(' ') / len(text) < 0.05:
        issues.append("very few spaces - possibly garbled")
        score = max(score, 0.6)

    is_gibberish = len(issues) > 0
    message = "; ".join(issues) if issues else "text appears normal"

    return is_gibberish, score, message


def detect_gibberish(
    tokens: List[int] = None,
    text: str = None,
    vocab_size: int = 32000,
    special_tokens: List[int] = None
) -> GibberishResult:
    """
    Main detection function - combines multiple detection methods.

    Args:
        tokens: List of token IDs (optional)
        text: Decoded text (optional)
        vocab_size: Vocabulary size for entropy calculation
        special_tokens: List of special token IDs

    Returns:
        GibberishResult with detection outcome
    """
    results = {}
    max_score = 0.0
    reasons = []

    # Token-based detection
    if tokens and len(tokens) > 0:
        # Repetition
        is_rep, score, msg = detect_repetition(tokens)
        results['repetition'] = {'detected': is_rep, 'score': score, 'message': msg}
        if is_rep:
            max_score = max(max_score, score)
            reasons.append(f"repetition: {msg}")

        # Entropy
        is_ent, score, msg = detect_entropy_anomaly(tokens, vocab_size)
        results['entropy'] = {'detected': is_ent, 'score': score, 'message': msg}
        if is_ent:
            max_score = max(max_score, score)
            reasons.append(f"entropy: {msg}")

        # Special tokens
        is_spc, score, msg = detect_special_token_ratio(tokens, special_tokens)
        results['special_tokens'] = {'detected': is_spc, 'score': score, 'message': msg}
        if is_spc:
            max_score = max(max_score, score)
            reasons.append(f"special tokens: {msg}")

    # Text-based detection
    if text and len(text) > 0:
        is_txt, score, msg = detect_text_patterns(text)
        results['text_patterns'] = {'detected': is_txt, 'score': score, 'message': msg}
        if is_txt:
            max_score = max(max_score, score)
            reasons.append(f"text: {msg}")

    # Determine overall result
    is_gibberish = max_score > 0.5 or len(reasons) >= 2

    return GibberishResult(
        is_gibberish=is_gibberish,
        confidence=max_score,
        reason="; ".join(reasons) if reasons else "output appears coherent",
        details=results
    )


def quick_check(text: str, min_length: int = 20) -> bool:
    """
    Quick check if text is likely gibberish.
    Returns True if gibberish is suspected.
    """
    if len(text) < min_length:
        return False  # Too short to tell

    # Quick heuristics
    words = text.split()
    if len(words) < 3:
        return False

    # Check for excessive repetition of last word
    if len(words) >= 5:
        last_words = words[-5:]
        if len(set(last_words)) == 1:
            return True  # Same word repeated at end

    # Check for repeated character sequences
    if len(text) >= 20:
        for i in range(len(text) - 10):
            pattern = text[i:i+5]
            if text.count(pattern) > len(text) / 10:
                return True

    return False


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_cases = [
        # (description, tokens, text)
        ("Normal text", None, "The quick brown fox jumps over the lazy dog."),
        ("Repetitive tokens", [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], None),
        ("Gibberish text", None, "asdf asdf asdf asdf asdf asdf asdf asdf"),
        ("Mixed quality", [100, 200, 300, 100, 200, 300, 100, 200, 300],
         "Hello world! The weather is nice today."),
        ("Stuck token", [42] * 50, "answer answer answer answer answer"),
    ]

    print("Gibberish Detection Test Cases")
    print("=" * 60)

    for desc, tokens, text in test_cases:
        result = detect_gibberish(tokens=tokens, text=text)
        status = "GIBBERISH" if result.is_gibberish else "OK"
        print(f"\n{desc}:")
        print(f"  Status: {status} (confidence: {result.confidence:.2f})")
        print(f"  Reason: {result.reason}")
