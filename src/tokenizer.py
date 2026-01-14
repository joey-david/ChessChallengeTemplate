"""
Tokenizers for the Chess Challenge.

- ChessTokenizer: byte-level BPE tokenizer for extended UCI move strings
- ChessSquaresTokenizer: square-level tokenizer with a fixed vocab
"""

from __future__ import annotations

import os
from typing import Dict, Iterator, List, Optional

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class ChessTokenizer(PreTrainedTokenizerFast):
    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {"tokenizer_file": "tokenizer.json"}

    PAD_TOKEN = "[PAD]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    UNK_TOKEN = "[UNK]"

    def __init__(
        self,
        tokenizer_file: Optional[str] = None,
        tokenizer_object: Optional[Tokenizer] = None,
        **kwargs,
    ):
        kwargs.setdefault("pad_token", self.PAD_TOKEN)
        kwargs.setdefault("bos_token", self.BOS_TOKEN)
        kwargs.setdefault("eos_token", self.EOS_TOKEN)
        kwargs.setdefault("unk_token", self.UNK_TOKEN)
        super().__init__(
            tokenizer_file=tokenizer_file,
            tokenizer_object=tokenizer_object,
            **kwargs,
        )

    @classmethod
    def _build_bpe_from_iterator(
        cls,
        iterator: Iterator[str],
        vocab_size: int,
        min_frequency: int,
    ) -> Tokenizer:
        tokenizer = Tokenizer(BPE(unk_token=cls.UNK_TOKEN))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=[cls.PAD_TOKEN, cls.BOS_TOKEN, cls.EOS_TOKEN, cls.UNK_TOKEN],
        )
        tokenizer.train_from_iterator(iterator, trainer=trainer)
        return tokenizer

    @classmethod
    def build_vocab_from_iterator(
        cls,
        iterator: Iterator[str],
        vocab_size: int = 1000,
        min_frequency: int = 2,
    ) -> "ChessTokenizer":
        tokenizer = cls._build_bpe_from_iterator(iterator, vocab_size, min_frequency)
        return cls(tokenizer_object=tokenizer)

    @classmethod
    def build_vocab_from_dataset(
        cls,
        dataset_name: str = "dlouapre/lichess_2025-01_1M",
        split: str = "train",
        column: str = "text",
        vocab_size: int = 1000,
        min_frequency: int = 2,
        max_samples: Optional[int] = None,
    ) -> "ChessTokenizer":
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, split=split)
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        def game_iterator() -> Iterator[str]:
            for example in dataset:
                yield example[column]

        tokenizer = cls._build_bpe_from_iterator(game_iterator(), vocab_size, min_frequency)
        return cls(tokenizer_object=tokenizer)

    def get_vocab(self) -> Dict[str, int]:
        return super().get_vocab()

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None,
    ) -> tuple:
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        tokenizer_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "tokenizer.json",
        )
        self.backend_tokenizer.save(tokenizer_file)
        return (tokenizer_file,)


class ChessSquaresTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {"vocab_file": "vocab.json"}

    PAD_TOKEN = "[PAD]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    UNK_TOKEN = "[UNK]"

    is_square_tokenizer = True

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        vocab: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        self._pad_token = self.PAD_TOKEN
        self._bos_token = self.BOS_TOKEN
        self._eos_token = self.EOS_TOKEN
        self._unk_token = self.UNK_TOKEN

        kwargs.pop("pad_token", None)
        kwargs.pop("bos_token", None)
        kwargs.pop("eos_token", None)
        kwargs.pop("unk_token", None)

        if vocab is not None:
            self._vocab = vocab
        elif vocab_file is not None and os.path.exists(vocab_file):
            import json
            with open(vocab_file, "r", encoding="utf-8") as f:
                self._vocab = json.load(f)
        else:
            self._vocab = self._create_default_vocab()

        self._ids_to_tokens = {v: k for k, v in self._vocab.items()}

        super().__init__(
            pad_token=self._pad_token,
            bos_token=self._bos_token,
            eos_token=self._eos_token,
            unk_token=self._unk_token,
            **kwargs,
        )

    def _create_default_vocab(self) -> Dict[str, int]:
        special_tokens = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        squares = [f"{file}{rank}" for rank in "12345678" for file in "abcdefgh"]
        promotions = ["q", "r", "b", "n"]
        vocab_tokens = special_tokens + squares + promotions
        return {token: idx for idx, token in enumerate(vocab_tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._vocab)

    def _tokenize(self, text: str) -> List[str]:
        return text.strip().split()

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab.get(token, self._vocab.get(self.UNK_TOKEN, 0))

    def _convert_id_to_token(self, index: int) -> str:
        return self._ids_to_tokens.get(index, self.UNK_TOKEN)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        special = {self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN}
        return " ".join(t for t in tokens if t not in special)

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None,
    ) -> tuple:
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        filename = (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        vocab_file = os.path.join(save_directory, filename)
        import json
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=True)
        return (vocab_file,)


def count_vocab_from_dataset(
    dataset_name: str = "dlouapre/lichess_2025-01_1M",
    split: str = "train",
    column: str = "text",
    max_samples: Optional[int] = 10000,
) -> Dict[str, int]:
    from collections import Counter
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    token_counts = Counter()
    for example in dataset:
        token_counts.update(example[column].strip().split())

    return dict(token_counts)
