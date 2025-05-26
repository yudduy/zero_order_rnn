"""
Tokenizers
"""
import string
from typing import List

# Optional tiktoken
try:
    import tiktoken
    _TIK_OK = True
except ImportError:
    _TIK_OK = False


class CharTokenizer:
    """Simple printable-ascii tokenizer (id-00 = unknown)."""
    def __init__(self) -> None:
        chars               = string.printable
        self.char_to_id     = {c: i for i, c in enumerate(chars)}
        self.id_to_char     = {i: c for c, i in self.char_to_id.items()}
        self.vocab_list     = chars
        self.vocab_size: int = len(chars)

    def encode(self, txt: str) -> List[int]:
        return [self.char_to_id.get(c, 0) for c in txt]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.id_to_char.get(i, '') for i in ids)


class NumericTokenizer:
    """
    Digits + arithmetic symbols – used by the ‘add’ task.
    """
    def __init__(self, max_num: int = 120) -> None:
        vocab               = "0123456789+-*/=()[],. "
        self.token_to_id    = {c: i for i, c in enumerate(vocab)}
        self.id_to_token    = {i: c for c, i in self.token_to_id.items()}
        self.vocab_size: int = len(vocab)
        self.max_num        = max_num

    def encode(self, txt: str):
        return [self.token_to_id.get(c, 0) for c in txt]

    def decode(self, ids):
        return ''.join(self.id_to_token.get(i, '') for i in ids)

    # helper used by tasks.add
    def encode_number(self, num: int):
        return self.encode(str(num))


def get_tiktoken(enc_name: str = "cl100k_base"):
    """
    Thin wrapper so the rest of the code never needs to import `tiktoken`
    directly.
    """
    if not _TIK_OK:
        raise ImportError("tiktoken not installed")
    enc = tiktoken.get_encoding(enc_name)

    class _TokWrap:
        def __init__(self, enc):
            self.enc = enc
            self.vocab_size = enc.n_vocab

        def encode(self, txt):
            return self.enc.encode(txt)

        def decode(self, ids):
            return self.enc.decode(ids)

    return _TokWrap(enc)
