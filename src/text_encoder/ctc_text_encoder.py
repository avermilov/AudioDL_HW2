import re
from string import ascii_lowercase

import kenlm
import torch
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel

from src.utils.io_utils import read_json


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(
        self,
        bpe_vocab_path: str = None,
        lm_path: str = None,
        lm_vocab_path: str = None,
        **kwargs,
    ):
        assert (lm_path is None) == (
            lm_vocab_path is None
        ), "If using LM, specify both lm_path and lm_vocab_path"
        self.lm_path = lm_path
        self.lm_vocab_path = lm_vocab_path
        self.bpe_vocab_path = bpe_vocab_path

        self.alphabet = (
            list(read_json(bpe_vocab_path).keys())
            if bpe_vocab_path
            else list(ascii_lowercase + " ")
        )
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.decoder_no_lm = BeamSearchDecoderCTC(
            Alphabet(self.vocab, False), language_model=None
        )
        if self.lm_path:
            kenlm_model = kenlm.Model(lm_path)
            with open(lm_vocab_path, "r") as f:
                unigrams = [t.lower() for t in f.read().strip().split("\n")]
            lm = LanguageModel(
                kenlm_model,
                unigrams,
                alpha=0.73,
                beta=1.37,
                unk_score_offset=-10.0,
                score_boundary=True,
            )
            self.decoder_lm = BeamSearchDecoderCTC(
                Alphabet(self.vocab, False), language_model=lm
            )

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert isinstance(item, int), "Index must be an integer."
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        decoded = []
        last_char_ind = self.char2ind[self.EMPTY_TOK]
        for ind in inds:
            if ind != last_char_ind and ind != self.char2ind[self.EMPTY_TOK]:
                decoded.append(self.ind2char[ind])
            last_char_ind = ind
        return "".join(decoded)

    def ctc_beam_search(
        self, probs, beam_size=100, use_lm=False
    ) -> list[dict[str, float]]:
        if use_lm:
            assert hasattr(self, "decoder_lm"), "No LM specified"
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        hyp = (
            self.decoder_lm.decode(probs, beam_size)
            if use_lm
            else self.decoder_no_lm.decode(probs, beam_size)
        )
        return [
            {
                "hyp": hyp,
                "prob": 1.0,
            }
        ]

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
