"""ClinicalShield Module 2 — PubMedBERT adversarial detection.

Wraps the fine-tuned classifier so the pipeline has one entry point rather than
inference code duplicated per notebook. Two calibration details are enforced
here because both cost real accuracy when they drifted during development:

``max_length`` defaults to 512, matching the training length. A value of 256
truncated a quarter of each 250-word chunk, so payloads injected into the tail
were never seen: document-level detection read 60.7% against 94.27% measured at
training time.

Module 1 ingestion must be applied BEFORE scoring. The classifier was trained on
decoded text, so passing raw text leaves encoded payloads obfuscated and costs
roughly eighteen points of detection.
"""

import numpy as np

DEFAULT_MAX_LENGTH = 512
DEFAULT_THRESHOLD = 0.50


class Module2Detector:
    """Detection over one or more fine-tuned checkpoints.

    Passing several checkpoints enables the seed ensemble, which lifted recall on
    externally authored attacks from 44.5% to 96.0% when combined with an
    adaptive threshold — at no training cost, since the checkpoints already
    exist.
    """

    def __init__(self, checkpoints, device=None, max_length=DEFAULT_MAX_LENGTH):
        import torch
        from transformers import (AutoTokenizer,
                                  AutoModelForSequenceClassification)
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        if isinstance(checkpoints, (str, bytes)) or hasattr(checkpoints, "__fspath__"):
            checkpoints = [checkpoints]
        self.checkpoints = list(checkpoints)
        self._tok = AutoTokenizer.from_pretrained(str(self.checkpoints[0]))
        self._models = [
            AutoModelForSequenceClassification.from_pretrained(str(c))
            .to(self.device).eval() for c in self.checkpoints]

    def scores(self, texts, batch_size=64, how="mean"):
        """P(adversarial) per text, pooled across checkpoints.

        ``how`` is "mean", "max" or "rank". Mean performed best on external
        attacks; rank is robust to per-checkpoint calibration differences.
        """
        torch = self._torch
        per_model = []
        with torch.no_grad():
            for mdl in self._models:
                out = []
                for i in range(0, len(texts), batch_size):
                    enc = self._tok(list(texts[i:i+batch_size]), truncation=True,
                                    padding=True, max_length=self.max_length,
                                    return_tensors="pt").to(self.device)
                    logits = mdl(**enc).logits
                    out.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
                per_model.append(np.asarray(out))
        A = np.vstack(per_model)
        if how == "mean":
            return A.mean(axis=0)
        if how == "max":
            return A.max(axis=0)
        if how == "rank":
            R = np.vstack([np.argsort(np.argsort(a)) / max(len(a) - 1, 1)
                           for a in A])
            return R.mean(axis=0)
        raise ValueError("how must be mean, max or rank")

    def predict(self, texts, threshold=DEFAULT_THRESHOLD, **kw):
        return (self.scores(texts, **kw) >= threshold).astype(int)

    def sentence_scores(self, text, segmenter, batch_size=64):
        """P(adversarial) per sentence, for Module 4.

        Judging sentences individually is what lets sanitisation excise the
        payload rather than the document: at document level a payload is roughly
        6% of the input, and alone it is all of it.
        """
        segs = [seg for _s0, _s1, seg in segmenter(text)]
        if not segs:
            return {}
        return dict(zip(segs, self.scores(segs, batch_size=batch_size)))


def adaptive_threshold(benign_scores, target_fpr=0.05):
    """Threshold at the (1 - target_fpr) quantile of observed benign scores.

    This fixes the false-positive rate rather than the threshold value, so it
    travels with a shifted score distribution. A fixed 0.5 cutoff is calibrated
    to the training phrasing: on externally authored attacks every score fell,
    and recall collapsed from 94% to 45% while AUC fell only 3.6 points — the
    ranking held, the cutoff did not. A deployed system sees benign documents
    continuously and can estimate this quantile online, without labels.
    """
    b = np.asarray(benign_scores, dtype=float)
    if b.size == 0:
        return DEFAULT_THRESHOLD
    return float(np.quantile(b, 1.0 - target_fpr))
