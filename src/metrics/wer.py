from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer
from src.text_encoder import CTCTextEncoder

# TODO beam search / LM versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class BeamSearchWERMetric(BaseMetric):
    def __init__(
        self,
        text_encoder: CTCTextEncoder,
        use_lm: bool = False,
        beam_size=10,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.use_lm = use_lm
        self.beam_size = beam_size

    def __call__(self, probs: Tensor, probs_length: Tensor, text: List[str], **kwargs):
        wers = []
        probs_arr = probs.cpu().detach().numpy()
        lengths = probs_length.detach().numpy()
        for prob, length, target_text in zip(probs_arr, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_beam_search(
                prob[:length], beam_size=self.beam_size, use_lm=self.use_lm
            )[0]["hyp"]
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)
