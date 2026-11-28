# Karpo Decision 0.8B generated readout

The Karpo circuit consumes final marker rows widened to FP32 and the number of
allowed options per question. It projects each row with the released FP32
readout and applies softmax to the allowed prefix. Columns keep their original
code indices. Inactive probability columns are zero. The caller associates
each row and column with the publisher encoder's question and option identities.

The released-weight acceptance on Ryzen used a two-question returns-policy
request with 2 and 3 options. Generated C matched the publisher's PyTorch
readout on identical marker rows with maximum absolute errors of
`9.5367432e-7` for raw scores and `3.7252903e-9` for probabilities. The test
bounds are `3e-5` and `1e-6` respectively. The capture uses the publisher
encoder and PyTorch backbone. It certifies the generated head on supplied
features. Connected backbone parity and readout training remain open.

Reproduce the acceptance with the pinned
[checkpoint](https://huggingface.co/MachinePulseAI/karpo-decision-0.8B/tree/fae75d7bf93fc7e5c37b1ff631b64e9474e4d797)
and [publisher checkout](https://github.com/machinepulse-ai/karpo-decision/tree/e5826d41e36c58e9e0829e4329c66c955903ddcd).
Install the publisher's dependencies in the oracle environment. The capture
command verifies the checkpoint hashes and requires a clean publisher checkout.

```bash
python -m version.v8.decision.capture_karpo_readout_reference \
  --checkpoint /path/to/karpo --publisher /path/to/publisher --output /tmp/karpo-oracle

CK_KARPO_READOUT_CHECKPOINT=/path/to/karpo \
CK_KARPO_READOUT_REFERENCE=/tmp/karpo-oracle \
  python -m pytest -q -s tests/test_v8_karpo_readout.py
```

Without asset paths, nightly runs the generated synthetic regression and
explicitly skips the released-weight case.
