"""
Custom Chess Tokenizer for the Chess Challenge.

This tokenizer supports move-level tokens by default, with an optional
structured split mode for experiments.

The dataset format uses:
- W/B prefix for White/Black
- Piece letter: P=Pawn, N=Knight, B=Bishop, R=Rook, Q=Queen, K=King
- Source and destination squares (e.g., e2e4)
- Promotions: =Q, =R, =B, =N
- Special suffixes: (x)=capture, (+)=check, (+*)=checkmate, (o)/(O)=castling
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer


class ChessTokenizer(PreTrainedTokenizer):
    """
    A custom tokenizer for chess moves using extended UCI notation.
    
    This tokenizer maps each chess move to a token by default. In split mode,
    each move is encoded as a fixed sequence of structured sub-tokens.
    
    Example:
        >>> tokenizer = ChessTokenizer()
        >>> tokenizer.encode("WPe2e4 BPe7e5")
        [1, 42, 87, 2]  # [BOS, WPe2e4, BPe7e5, EOS]
    """
    
    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {"vocab_file": "vocab.json"}
    
    # Special tokens
    PAD_TOKEN = "[PAD]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    UNK_TOKEN = "[UNK]"

    # Structured move tokens
    COLOR_TOKENS = ["C_W", "C_B"]
    PIECE_TOKENS = ["PIECE_P", "PIECE_N", "PIECE_B", "PIECE_R", "PIECE_Q", "PIECE_K"]
    SQUARE_TOKENS = [
        f"SQ_{file}{rank}" for file in "abcdefgh" for rank in "12345678"
    ]
    PROMO_TOKENS = ["PROMO_NONE", "PROMO_Q", "PROMO_R", "PROMO_B", "PROMO_N"]
    SUFFIX_TOKENS = [
        "SUF_NONE",
        "SUF_X",
        "SUF_PLUS",
        "SUF_MATE",
        "SUF_XPLUS",
        "SUF_XMATE",
        "SUF_O",
        "SUF_OO",
    ]

    _COLOR_TOKEN_MAP = {"W": "C_W", "B": "C_B"}
    _PIECE_TOKEN_MAP = {
        "P": "PIECE_P",
        "N": "PIECE_N",
        "B": "PIECE_B",
        "R": "PIECE_R",
        "Q": "PIECE_Q",
        "K": "PIECE_K",
    }
    _PROMO_TOKEN_MAP = {
        None: "PROMO_NONE",
        "Q": "PROMO_Q",
        "R": "PROMO_R",
        "B": "PROMO_B",
        "N": "PROMO_N",
    }
    _SUFFIX_TOKEN_MAP = {
        None: "SUF_NONE",
        "": "SUF_NONE",
        "x": "SUF_X",
        "+": "SUF_PLUS",
        "+*": "SUF_MATE",
        "x+": "SUF_XPLUS",
        "x+*": "SUF_XMATE",
        "o": "SUF_O",
        "O": "SUF_OO",
    }

    _TOKEN_TO_COLOR = {v: k for k, v in _COLOR_TOKEN_MAP.items()}
    _TOKEN_TO_PIECE = {v: k for k, v in _PIECE_TOKEN_MAP.items()}
    _TOKEN_TO_PROMO = {
        "PROMO_NONE": "",
        "PROMO_Q": "=Q",
        "PROMO_R": "=R",
        "PROMO_B": "=B",
        "PROMO_N": "=N",
    }
    _TOKEN_TO_SUFFIX = {
        "SUF_NONE": "",
        "SUF_X": "(x)",
        "SUF_PLUS": "(+)",
        "SUF_MATE": "(+*)",
        "SUF_XPLUS": "(x+)",
        "SUF_XMATE": "(x+*)",
        "SUF_O": "(o)",
        "SUF_OO": "(O)",
    }
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        vocab: Optional[Dict[str, int]] = None,
        split_moves: bool = False,
        **kwargs,
    ):
        """
        Initialize the chess tokenizer.
        
        Args:
            vocab_file: Path to a JSON file containing the vocabulary mapping.
            vocab: Dictionary mapping tokens to IDs (alternative to vocab_file).
            **kwargs: Additional arguments passed to PreTrainedTokenizer.
        """
        # Initialize special tokens
        self._pad_token = self.PAD_TOKEN
        self._bos_token = self.BOS_TOKEN
        self._eos_token = self.EOS_TOKEN
        self._unk_token = self.UNK_TOKEN
        self._split_moves = split_moves

        # Remove any duplicate special-token entries passed through kwargs
        # to avoid "multiple values for keyword" errors when loading from disk.
        kwargs.pop("pad_token", None)
        kwargs.pop("bos_token", None)
        kwargs.pop("eos_token", None)
        kwargs.pop("unk_token", None)
        kwargs.pop("split_moves", None)
        
        # Load or create vocabulary
        if vocab is not None:
            self._vocab = vocab
        elif vocab_file is not None and os.path.exists(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                self._vocab = json.load(f)
        else:
            # Create a minimal vocabulary with just special tokens
            # The full vocabulary should be built from the dataset
            self._vocab = self._create_default_vocab()
        
        # Create reverse mapping
        self._ids_to_tokens = {v: k for k, v in self._vocab.items()}
        
        # Call parent init AFTER setting up vocab
        super().__init__(
            pad_token=self._pad_token,
            bos_token=self._bos_token,
            eos_token=self._eos_token,
            unk_token=self._unk_token,
            split_moves=split_moves,
            **kwargs,
        )
    
    def _create_default_vocab(self) -> Dict[str, int]:
        """
        Create a minimal default vocabulary with just special tokens.
        
        For the full vocabulary, use `build_vocab_from_dataset()`.
        This minimal vocab is just a placeholder - you should build from data.
        """
        special_tokens = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        vocab = {token: idx for idx, token in enumerate(special_tokens)}
        return vocab

    @classmethod
    def _get_base_tokens(cls) -> List[str]:
        return (
            cls.COLOR_TOKENS
            + cls.PIECE_TOKENS
            + cls.SQUARE_TOKENS
            + cls.PROMO_TOKENS
            + cls.SUFFIX_TOKENS
        )

    @classmethod
    def _split_move(
        cls,
        move: str,
    ) -> Optional[Tuple[str, str, str, str, Optional[str], Optional[str]]]:
        if len(move) < 6:
            return None

        color = move[0]
        piece = move[1]
        from_sq = move[2:4]
        to_sq = move[4:6]
        rest = move[6:]

        if color not in cls._COLOR_TOKEN_MAP:
            return None
        if piece not in cls._PIECE_TOKEN_MAP:
            return None
        if f"SQ_{from_sq}" not in cls.SQUARE_TOKENS:
            return None
        if f"SQ_{to_sq}" not in cls.SQUARE_TOKENS:
            return None

        promo = None
        suffix = None
        if rest:
            if rest.startswith("="):
                if len(rest) < 2:
                    return None
                promo = rest[1]
                rest = rest[2:]
            if rest:
                if rest.startswith("(") and rest.endswith(")"):
                    suffix = rest[1:-1]
                else:
                    return None

        if promo is not None and promo not in cls._PROMO_TOKEN_MAP:
            return None
        if suffix is not None and suffix not in cls._SUFFIX_TOKEN_MAP:
            return None

        return color, piece, from_sq, to_sq, promo, suffix
    
    @classmethod
    def build_vocab_from_iterator(
        cls,
        iterator,
        min_frequency: int = 1,
        split_moves: bool = False,
    ) -> "ChessTokenizer":
        """
        Build a tokenizer vocabulary from an iterator of game strings.
        
        Args:
            iterator: An iterator yielding game strings (space-separated moves).
            min_frequency: Minimum frequency for a token to be included.
        
        Returns:
            A ChessTokenizer with the built vocabulary.
        """
        from collections import Counter
        
        token_counts = Counter()

        if split_moves:
            base_tokens = cls._get_base_tokens()
            for game in iterator:
                moves = game.strip().split()
                for move in moves:
                    move_parts = cls._split_move(move)
                    if move_parts is None:
                        token_counts.update([cls.UNK_TOKEN])
                        continue
                    color, piece, from_sq, to_sq, promo, suffix = move_parts
                    token_counts.update([
                        cls._COLOR_TOKEN_MAP[color],
                        cls._PIECE_TOKEN_MAP[piece],
                        f"SQ_{from_sq}",
                        f"SQ_{to_sq}",
                        cls._PROMO_TOKEN_MAP[promo],
                        cls._SUFFIX_TOKEN_MAP[suffix],
                    ])
        else:
            for game in iterator:
                moves = game.strip().split()
                token_counts.update(moves)
        
        # Filter by frequency
        tokens = [
            token for token, count in token_counts.items()
            if count >= min_frequency
        ]

        # Build vocabulary
        special_tokens = [cls.PAD_TOKEN, cls.BOS_TOKEN, cls.EOS_TOKEN, cls.UNK_TOKEN]
        if split_moves:
            base_tokens = cls._get_base_tokens()
            extra_tokens = sorted(t for t in tokens if t not in base_tokens)
            full_tokens = base_tokens + extra_tokens
            vocab_tokens = special_tokens + full_tokens
        else:
            vocab_tokens = special_tokens + sorted(tokens)
        vocab = {token: idx for idx, token in enumerate(vocab_tokens)}
        
        return cls(vocab=vocab, split_moves=split_moves)
    
    @classmethod
    def build_vocab_from_dataset(
        cls,
        dataset_name: str = "dlouapre/lichess_2025-01_1M",
        split: str = "train",
        column: str = "text",
        min_frequency: int = 500,
        max_samples: Optional[int] = 100000,
        split_moves: bool = False,
    ) -> "ChessTokenizer":
        """
        Build a tokenizer vocabulary from a Hugging Face dataset.
        
        Args:
            dataset_name: Name of the dataset on Hugging Face Hub.
            split: Dataset split to use.
            column: Column containing the game strings.
            min_frequency: Minimum frequency for a token to be included (default: 500).
            max_samples: Maximum number of samples to process (default: 100k).
        
        Returns:
            A ChessTokenizer with the built vocabulary.
        """
        from datasets import load_dataset
        
        dataset = load_dataset(dataset_name, split=split)
        
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        def game_iterator():
            for example in dataset:
                yield example[column]
        
        return cls.build_vocab_from_iterator(
            game_iterator(),
            min_frequency=min_frequency,
            split_moves=split_moves,
        )
    
    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self._vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary as a dictionary."""
        return dict(self._vocab)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize a string of moves into a list of tokens.
        
        Args:
            text: A string of space-separated moves.
        
        Returns:
            List of move tokens.
        """
        if not text:
            return []

        raw_tokens = text.strip().split()
        if not self._split_moves:
            return raw_tokens

        tokens: List[str] = []
        for raw in raw_tokens:
            if raw in {self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN}:
                tokens.append(raw)
                continue

            move_parts = self._split_move(raw)
            if move_parts is None:
                tokens.append(self.UNK_TOKEN)
                continue

            color, piece, from_sq, to_sq, promo, suffix = move_parts
            tokens.extend([
                self._COLOR_TOKEN_MAP[color],
                self._PIECE_TOKEN_MAP[piece],
                f"SQ_{from_sq}",
                f"SQ_{to_sq}",
                self._PROMO_TOKEN_MAP[promo],
                self._SUFFIX_TOKEN_MAP[suffix],
            ])

        return tokens
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token to its ID."""
        return self._vocab.get(token, self._vocab.get(self.UNK_TOKEN, 0))
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert an ID to its token."""
        return self._ids_to_tokens.get(index, self.UNK_TOKEN)
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert a list of tokens back to a string."""
        special = {self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN}
        filtered = [t for t in tokens if t not in special]

        if not self._split_moves:
            return " ".join(filtered)

        moves: List[str] = []
        i = 0
        while i + 5 < len(filtered):
            color_tok = filtered[i]
            piece_tok = filtered[i + 1]
            from_tok = filtered[i + 2]
            to_tok = filtered[i + 3]
            promo_tok = filtered[i + 4]
            suffix_tok = filtered[i + 5]

            if (
                color_tok not in self._TOKEN_TO_COLOR
                or piece_tok not in self._TOKEN_TO_PIECE
                or not from_tok.startswith("SQ_")
                or not to_tok.startswith("SQ_")
                or promo_tok not in self._TOKEN_TO_PROMO
                or suffix_tok not in self._TOKEN_TO_SUFFIX
            ):
                i += 1
                continue

            color = self._TOKEN_TO_COLOR[color_tok]
            piece = self._TOKEN_TO_PIECE[piece_tok]
            from_sq = from_tok.replace("SQ_", "", 1)
            to_sq = to_tok.replace("SQ_", "", 1)
            promo = self._TOKEN_TO_PROMO[promo_tok]
            suffix = self._TOKEN_TO_SUFFIX[suffix_tok]

            moves.append(f"{color}{piece}{from_sq}{to_sq}{promo}{suffix}")
            i += 6

        return " ".join(moves)
    
    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None,
    ) -> tuple:
        """
        Save the vocabulary to a JSON file.
        
        Args:
            save_directory: Directory to save the vocabulary.
            filename_prefix: Optional prefix for the filename.
        
        Returns:
            Tuple containing the path to the saved vocabulary file.
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json",
        )
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)

        self._update_model_config_vocab(save_directory)
        
        return (vocab_file,)

    def _update_model_config_vocab(self, save_directory: str) -> None:
        config_path = os.path.join(save_directory, "config.json")
        if not os.path.exists(config_path):
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        max_id = max(self._vocab.values(), default=-1)
        id_to_token = [self.UNK_TOKEN] * (max_id + 1)
        for token, idx in self._vocab.items():
            if 0 <= idx <= max_id:
                id_to_token[idx] = token

        config_data["id_to_token"] = id_to_token

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
        except OSError:
            return


def count_vocab_from_dataset(
    dataset_name: str = "dlouapre/lichess_2025-01_1M",
    split: str = "train",
    column: str = "text",
    max_samples: Optional[int] = 10000,
) -> Dict[str, int]:
    """
    Count token frequencies in a dataset (useful for vocabulary analysis).
    
    Args:
        dataset_name: Name of the dataset on Hugging Face Hub.
        split: Dataset split to use.
        column: Column containing the game strings.
        max_samples: Maximum number of samples to process.
    
    Returns:
        Dictionary mapping tokens to their frequencies.
    """
    from collections import Counter
    from datasets import load_dataset
    
    dataset = load_dataset(dataset_name, split=split)
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    token_counts = Counter()
    
    for example in dataset:
        moves = example[column].strip().split()
        token_counts.update(moves)
    
    return dict(token_counts)
