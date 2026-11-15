"""Explicit sentence segmentation for long Ming trials, not whole-utterance parity."""
import re

def sentence_spans(text):
    if not isinstance(text,str) or not text.strip() or len(text)>100000:
        raise ValueError('expected bounded nonempty text')
    # Keep whitespace in source spans: joining their source slices is lossless.
    # Abbreviations are deliberately not guessed; clients may provide their own spans.
    ends=[m.end() for m in re.finditer(r'[.!?](?:[\"\'])?\s+(?=[A-Z0-9])',text)]
    bounds=[0,*ends,len(text)]
    return [(a,b) for a,b in zip(bounds,bounds[1:]) if a<b]

def validate_spans(text,spans):
    position=0
    for start,end in spans:
        if type(start) is not int or type(end) is not int or start!=position or not start<end<=len(text):
            raise ValueError('segments must partition the original text without gaps or overlaps')
        if not text[start:end].strip():raise ValueError('empty spoken segment')
        position=end
    if position!=len(text):raise ValueError('segments omit text')
