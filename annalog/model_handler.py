# smiles_model_handler.py (torchtext-free; uses *.nott.pkl vocabs)

import pickle
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from .model_files import seq2seq_attention, vocabulary


class _StoiDict(dict):
    """dict that mimics torchtext's stoi defaultdict behavior (missing -> unk index, and inserts key)."""
    def __init__(self, *args, unk_index: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._unk_index = int(unk_index)

    def __missing__(self, key):
        self[key] = self._unk_index
        return self._unk_index


class SimpleVocab:
    """Minimal vocab wrapper with .itos, .stoi (default-to-unk), and __len__."""
    def __init__(self, itos: Sequence[str], stoi: Dict[str, int], unk_token: str = "<unk>"):
        itos = list(itos)
        stoi = dict(stoi)

        if unk_token not in stoi:
            raise ValueError(f"unk_token '{unk_token}' not found in vocab.stoi")

        self.itos = itos
        self.unk_token = unk_token
        self.unk_index = int(stoi[unk_token])

        # mimic torchtext behavior: stoi[...] returns unk index for missing keys
        self.stoi = _StoiDict(stoi, unk_index=self.unk_index)

    def __len__(self) -> int:
        return len(self.itos)

    def lookup(self, token: str) -> int:
        return self.stoi[token]  # safe for OOV


class SimpleField:
    """
    Minimal torchtext.Field-like subset you used (validated in your parity tests):
      - preprocess(text) -> List[str] tokens
      - process(list_of_token_lists, device=...) -> LongTensor [B, L] (batch_first=True)
    """
    def __init__(
        self,
        tokenize: Callable[[str], List[str]],
        init_token: str = "<sos>",
        eos_token: str = "<eos>",
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        lower: bool = False,
        batch_first: bool = True,
    ):
        self.tokenize = tokenize
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.lower = lower
        self.batch_first = batch_first
        self.vocab: Optional[SimpleVocab] = None

    def preprocess(self, text: str) -> List[str]:
        if self.lower:
            text = text.lower()
        return list(self.tokenize(text))

    def _pad(self, batch_tokens: Sequence[Sequence[str]]) -> List[List[str]]:
        seqs: List[List[str]] = []
        for toks in batch_tokens:
            toks = list(toks)
            seqs.append([self.init_token] + toks + [self.eos_token])

        max_len = max((len(s) for s in seqs), default=0)
        return [s + [self.pad_token] * (max_len - len(s)) for s in seqs]

    def process(self, batch_tokens: Sequence[Sequence[str]], device: Optional[torch.device] = None) -> torch.Tensor:
        if self.vocab is None:
            raise RuntimeError("SimpleField.vocab must be set before calling process().")

        padded = self._pad(batch_tokens)
        ids = [[self.vocab.lookup(tok) for tok in seq] for seq in padded]
        t = torch.tensor(ids, dtype=torch.long, device=device)

        if not self.batch_first:
            t = t.t().contiguous()
        return t


def _load_nott_vocab(path: str) -> Tuple[List[str], Dict[str, int]]:
    """Load torchtext-free vocab pkl: {'itos':[...], 'stoi':{...}}"""
    with open(path, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict) or "itos" not in payload or "stoi" not in payload:
        raise RuntimeError(
            f"Vocab file '{path}' is not in expected torchtext-free format.\n"
            "Expected a dict with keys {'itos','stoi'} (e.g. *.nott.pkl)."
        )

    return list(payload["itos"]), dict(payload["stoi"])


def _require_specials(stoi: Dict[str, int], vocab_path: str):
    required = ["<unk>", "<pad>", "<sos>", "<eos>"]
    missing = [t for t in required if t not in stoi]
    if missing:
        raise RuntimeError(
            f"Vocab '{vocab_path}' is missing required special tokens: {missing}"
        )


class SMILESModelHandler:
    """
    Torchtext-free inference handler.
    Uses new torchtext-free vocab PKLs (exported from your old torchtext vocabs).
    """
    def __init__(self, src_vocab_path, trg_vocab_path, model_path, device="cuda", max_length=102):
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.use_masking = True

        tokenize = vocabulary.SMILESTokenizer().tokenize

        src_itos, src_stoi = _load_nott_vocab(src_vocab_path)
        trg_itos, trg_stoi = _load_nott_vocab(trg_vocab_path)

        _require_specials(src_stoi, src_vocab_path)
        _require_specials(trg_stoi, trg_vocab_path)

        self.SRC = SimpleField(tokenize=tokenize, init_token="<sos>", eos_token="<eos>",
                               pad_token="<pad>", unk_token="<unk>", lower=False, batch_first=True)
        self.TRG = SimpleField(tokenize=tokenize, init_token="<sos>", eos_token="<eos>",
                               pad_token="<pad>", unk_token="<unk>", lower=False, batch_first=True)

        self.SRC.vocab = SimpleVocab(src_itos, src_stoi, unk_token="<unk>")
        self.TRG.vocab = SimpleVocab(trg_itos, trg_stoi, unk_token="<unk>")

        # Must match training/checkpoint
        INPUT_DIM = len(self.SRC.vocab)
        OUTPUT_DIM = len(self.TRG.vocab)
        HID_DIM = 256
        ENC_LAYERS = 3
        DEC_LAYERS = 3
        ENC_HEADS = 8
        DEC_HEADS = 8
        ENC_PF_DIM = 512
        DEC_PF_DIM = 512
        ENC_DROPOUT = 0.1
        DEC_DROPOUT = 0.1

        enc = seq2seq_attention.Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, self.device, max_length)
        dec = seq2seq_attention.Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, self.device, max_length)

        SRC_PAD_IDX = self.SRC.vocab.stoi["<pad>"]
        TRG_PAD_IDX = self.TRG.vocab.stoi["<pad>"]

        self.model = seq2seq_attention.Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, self.device).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def get_model_and_fields(self):
        return self.SRC, self.TRG, self.model, self.device, self.use_masking