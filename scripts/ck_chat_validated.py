#!/usr/bin/env python3
"""
C-Kernel-Engine Chat Interface with Auto-Validation

This is a wrapper around ck_chat.py that adds automatic gibberish detection
and validation when output appears corrupted.

Usage:
    python scripts/ck_chat_validated.py --model-dir <dir> [--gguf <path>]

When gibberish is detected:
1. Stops generation
2. Runs staged validation (Stage 1, 2, 3)
3. Prints debugging instructions

This helps quickly identify kernel bugs after code changes.
"""

import sys
import os
from pathlib import Path

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "unittest"))

import argparse
from typing import List, Optional

# Import the original chat module
from ck_chat import CKModel, generate, sample_top_k, main as chat_main

# Import validation - add unittest path for proper imports
sys.path.insert(0, str(PROJECT_ROOT / "unittest"))
from validation.gibberish_detector import detect_gibberish, quick_check
from validation.auto_validate import AutoValidator

# Colors
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'


class ValidatedChatSession:
    """Chat session with gibberish detection and auto-validation."""

    def __init__(
        self,
        model_dir: str,
        gguf_path: Optional[str] = None,
        bump_path: Optional[str] = None,
        manifest_path: Optional[str] = None,
        check_every_n_tokens: int = 20,
        verbose: bool = False
    ):
        self.model_dir = Path(model_dir)
        self.gguf_path = gguf_path
        self.bump_path = bump_path or str(self.model_dir / "weights.bump")
        self.manifest_path = manifest_path or str(self.model_dir / "weights_manifest.json")
        self.check_every_n_tokens = check_every_n_tokens
        self.verbose = verbose

        # Initialize model
        self.model = CKModel(str(self.model_dir))
        self.model.load(gguf_path)

        # Initialize validator
        self.validator = AutoValidator(
            gguf_path=self.gguf_path or "",
            bump_path=self.bump_path,
            manifest_path=self.manifest_path,
            work_dir=str(self.model_dir),
            verbose=verbose
        )

        # Track generated tokens for detection
        self.generated_tokens: List[int] = []
        self.generated_text: str = ""
        self.gibberish_detected = False

    def generate_with_validation(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7
    ) -> str:
        """Generate text with gibberish detection."""
        self.generated_tokens = []
        self.generated_text = ""
        self.gibberish_detected = False

        # Tokenize prompt
        token_ids = self.model.encode(prompt)
        prompt_len = len(token_ids)

        if self.model.has_kv_decode and self.model.kv_cache_enable():
            # KV-cache path
            logits = self.model.prefill(token_ids)

            for i in range(max_tokens):
                next_token = sample_top_k(logits, k=40, temperature=temperature)

                if self.model.is_eos_token(next_token):
                    break

                self.generated_tokens.append(next_token)
                token_text = self.model.decode([next_token])
                self.generated_text += token_text
                print(token_text, end='', flush=True)

                # Periodic gibberish check
                if (i + 1) % self.check_every_n_tokens == 0:
                    if self._check_for_gibberish():
                        self._handle_gibberish()
                        return self.generated_text

                token_ids.append(next_token)
                if len(token_ids) >= self.model.context_window - 1:
                    break

                logits = self.model.decode_step(next_token)
        else:
            # Non-KV path
            for i in range(max_tokens):
                logits = self.model.forward(token_ids)
                next_token = sample_top_k(logits, k=40, temperature=temperature)

                if self.model.is_eos_token(next_token):
                    break

                self.generated_tokens.append(next_token)
                token_text = self.model.decode([next_token])
                self.generated_text += token_text
                print(token_text, end='', flush=True)

                # Periodic gibberish check
                if (i + 1) % self.check_every_n_tokens == 0:
                    if self._check_for_gibberish():
                        self._handle_gibberish()
                        return self.generated_text

                token_ids.append(next_token)
                if len(token_ids) >= self.model.context_window - 1:
                    break

        # Final check
        if len(self.generated_tokens) > 10:
            if self._check_for_gibberish():
                self._handle_gibberish()

        print()  # newline
        return self.generated_text

    def _check_for_gibberish(self) -> bool:
        """Check if current output is gibberish."""
        # Quick check on text
        if quick_check(self.generated_text):
            return True

        # Full check if we have enough tokens
        if len(self.generated_tokens) >= 10:
            result = detect_gibberish(
                tokens=self.generated_tokens,
                text=self.generated_text,
                vocab_size=self.model.vocab_size
            )
            if result.is_gibberish and result.confidence > 0.5:
                return True

        return False

    def _handle_gibberish(self):
        """Handle detected gibberish by running validation."""
        self.gibberish_detected = True
        print()  # newline

        # Run auto-validation
        if not self.validator.check_output(
            tokens=self.generated_tokens,
            text=self.generated_text,
            vocab_size=self.model.vocab_size
        ):
            self.validator.run_validation()
            self.validator.print_debug_instructions()

    def chat_loop(self, max_tokens: int = 100, temperature: float = 0.7):
        """Interactive chat loop with validation."""
        print(f"\n{CYAN}C-Kernel-Engine Chat (with auto-validation){RESET}")
        print(f"{CYAN}Type 'quit' to exit, 'validate' to run manual validation{RESET}\n")

        while True:
            try:
                user_input = input(f"{GREEN}You:{RESET} ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                break

            if user_input.lower() == 'validate':
                print(f"\n{CYAN}Running manual validation...{RESET}")
                self.validator.run_validation()
                continue

            print(f"{YELLOW}Assistant:{RESET} ", end='')
            response = self.generate_with_validation(
                user_input,
                max_tokens=max_tokens,
                temperature=temperature
            )

            if self.gibberish_detected:
                print(f"\n{RED}[Generation stopped due to gibberish]{RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="C-Kernel-Engine Chat with Auto-Validation"
    )
    parser.add_argument("--model-dir", required=True, help="Model directory")
    parser.add_argument("--gguf", help="GGUF model path")
    parser.add_argument("--bump", help="BUMP weights path")
    parser.add_argument("--manifest", help="Manifest path")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--prompt", help="Single prompt (non-interactive)")
    parser.add_argument("--check-every", type=int, default=20,
                       help="Check for gibberish every N tokens")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    session = ValidatedChatSession(
        model_dir=args.model_dir,
        gguf_path=args.gguf,
        bump_path=args.bump,
        manifest_path=args.manifest,
        check_every_n_tokens=args.check_every,
        verbose=args.verbose
    )

    if args.prompt:
        # Single prompt mode
        print(f"{YELLOW}Response:{RESET} ", end='')
        session.generate_with_validation(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
    else:
        # Interactive mode
        session.chat_loop(
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )


if __name__ == "__main__":
    main()
