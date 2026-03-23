import heapq
import math
from typing import List, Tuple

import kenlm
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# ---------------------------------------------------------------------------
# Provided utility — do NOT modify
# ---------------------------------------------------------------------------


def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


class Wav2Vec2Decoder:
    def __init__(
        self,
        model_name="facebook/wav2vec2-base-100h",
        lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
        beam_width=3,
        alpha=1.0,
        beta=1.0,
        temperature=1.0,
    ):
        """
        Args:
            model_name (str): Pretrained Wav2Vec2 model from HuggingFace.
            lm_model_path (str): Path to a KenLM .arpa/.arpa.gz model.
                Pass None to disable LM (Tasks 1–3).
            beam_width (int): Number of hypotheses kept during beam search.
            alpha (float): LM weight used in shallow fusion and rescoring.
                score = log_p_acoustic + alpha * log_p_lm + beta * num_words
            beta (float): Word insertion bonus (see above).
            temperature (float): Scales acoustic logits before softmax.
                T < 1 sharpens the distribution (model more confident).
                T > 1 flattens it (model less confident, giving LM more
                influence). T = 1.0 leaves logits unchanged.
        """
        # Interact with processor/model ONLY here and in decode() to obtain
        # logits — no further model calls are allowed anywhere else.
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None

    # -----------------------------------------------------------------------
    # Provided utility — do NOT modify
    # -----------------------------------------------------------------------

    def _ids_to_text(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs to a decoded string."""
        text = "".join(self.vocab[i] for i in token_ids)
        return text.replace(self.word_delimiter, " ").strip().lower()

    # -----------------------------------------------------------------------
    # Tasks 1–4: implement the methods below
    # -----------------------------------------------------------------------

    def greedy_decode(self, logits: torch.Tensor) -> str:
        """
        Perform greedy decoding (find best CTC path).

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V).

        Returns:
            str: Decoded transcript.
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        indices = log_probs.argmax(dim=1).tolist()
        pr = None
        out = []
        for ind in indices:
            if ind != pr:
                if ind != self.blank_token_id:
                    out.append(ind)
            pr = ind
        return self._ids_to_text(out)

    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        """
        Perform beam search decoding (no LM).

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size.
            return_beams (bool): Return all beam hypotheses for second-pass
                LM rescoring.

        Returns:
            Union[str, List[Tuple[List[int], float]]]:
                str - best decoded transcript (if return_beams=False).
                List[Tuple[List[int], float]] - list of (token_ids, log_prob)
                    tuples sorted best-first (if return_beams=True).
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        T, V = log_probs.shape

        # prefix -> (p_blank, p_non_blank)
        beams = {(): (0.0, float("-inf"))}

        for t in range(len(log_probs)):
            new_beams = {}

            for prefix, (p_b, p_nb) in beams.items():
                last = prefix[-1] if prefix else None

                for c in range(log_probs.shape[1]):
                    score = log_probs[t, c].item()

                    if c == self.blank_token_id:
                        # blank never changes the prefix
                        cur = new_beams.get(prefix, (float("-inf"), float("-inf")))
                        new_beams[prefix] = (_log_add(cur[0], _log_add(p_b, p_nb) + score), cur[1])

                    elif c == last:
                        # same token as last:
                        # non-blank path collapses → stays at same prefix
                        cur = new_beams.get(prefix, (float("-inf"), float("-inf")))
                        new_beams[prefix] = (cur[0], _log_add(cur[1], p_nb + score))
                        # blank path separates → extends to new prefix
                        new_prefix = prefix + (c,)
                        cur2 = new_beams.get(new_prefix, (float("-inf"), float("-inf")))
                        new_beams[new_prefix] = (cur2[0], _log_add(cur2[1], p_b + score))

                    else:
                        # different token → always extends prefix
                        new_prefix = prefix + (c,)
                        cur = new_beams.get(new_prefix, (float("-inf"), float("-inf")))
                        new_beams[new_prefix] = (
                            cur[0],
                            _log_add(cur[1], _log_add(p_b, p_nb) + score),
                        )

            # prune to top beam_width
            beams = dict(
                heapq.nlargest(self.beam_width, new_beams.items(), key=lambda x: _log_add(x[1][0], x[1][1]))
            )

        sorted_beams = heapq.nlargest(len(beams), beams.items(), key=lambda x: _log_add(x[1][0], x[1][1]))

        if return_beams:
            return [(list(prefix), _log_add(p_b, p_nb)) for prefix, (p_b, p_nb) in sorted_beams]

        return self._ids_to_text(list(sorted_beams[0][0]))

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        """
        Perform beam search decoding with shallow LM fusion.

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size.

        Returns:
            str: Decoded transcript.
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM shallow fusion")

        log_probs = torch.log_softmax(logits, dim=-1)
        T, V = log_probs.shape

        delimiter_id = next(i for i, c in self.vocab.items() if c == self.word_delimiter)

        def lm_score_for(prefix):
            text = self._ids_to_text(list(prefix))
            if not text.strip():
                return 0.0, 0
            return self.lm_model.score(text, bos=True, eos=False) * math.log(10), len(text.split())

        # prefix -> (p_blank, p_non_blank, lm_score, num_words)
        beams = {(): (0.0, float("-inf"), 0.0, 0)}

        for t in range(T):
            new_beams = {}

            for prefix, (p_b, p_nb, lm_sc, n_words) in beams.items():
                last = prefix[-1] if prefix else None

                for c in range(V):
                    ac = log_probs[t, c].item()

                    if c == self.blank_token_id:
                        cur = new_beams.get(prefix, (float("-inf"), float("-inf"), lm_sc, n_words))
                        new_beams[prefix] = (
                            _log_add(cur[0], _log_add(p_b, p_nb) + ac),
                            cur[1],
                            lm_sc,
                            n_words,
                        )

                    elif c == last:
                        # non-blank collapses → same prefix
                        cur = new_beams.get(prefix, (float("-inf"), float("-inf"), lm_sc, n_words))
                        new_beams[prefix] = (cur[0], _log_add(cur[1], p_nb + ac), lm_sc, n_words)
                        # blank-separated → new token appended
                        new_prefix = prefix + (c,)
                        new_lm, new_nw = (
                            lm_score_for(new_prefix) if c == delimiter_id else (lm_sc, n_words)
                        )
                        cur2 = new_beams.get(
                            new_prefix, (float("-inf"), float("-inf"), new_lm, new_nw)
                        )
                        new_beams[new_prefix] = (
                            cur2[0],
                            _log_add(cur2[1], p_b + ac),
                            new_lm,
                            new_nw,
                        )

                    else:
                        new_prefix = prefix + (c,)
                        new_lm, new_nw = (
                            lm_score_for(new_prefix) if c == delimiter_id else (lm_sc, n_words)
                        )
                        cur = new_beams.get(
                            new_prefix, (float("-inf"), float("-inf"), new_lm, new_nw)
                        )
                        new_beams[new_prefix] = (
                            cur[0],
                            _log_add(cur[1], _log_add(p_b, p_nb) + ac),
                            new_lm,
                            new_nw,
                        )

            def combined(item):
                _, (pb, pnb, lm, nw) = item
                return _log_add(pb, pnb) + self.alpha * lm + self.beta * nw

            beams = dict(heapq.nlargest(self.beam_width, new_beams.items(), key=combined))

        best = max(
            beams.items(),
            key=lambda x: _log_add(x[1][0], x[1][1]) + self.alpha * x[1][2] + self.beta * x[1][3],
        )
        return self._ids_to_text(list(best[0]))

    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        """
        Perform second-pass LM rescoring on beam search outputs.

        Args:
            beams (List[Tuple[List[int], float]]): List of (token_ids, log_prob)
                tuples from beam_search_decode(logits, return_beams=True).

        Returns:
            str: Best rescored transcript.
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM rescoring")

        def score(beam):
            token_ids, log_p_acoustic = beam
            text = self._ids_to_text(token_ids)
            num_words = len(text.split())
            log_p_lm = self.lm_model.score(text, bos=True, eos=True) * math.log(10)
            return log_p_acoustic + self.alpha * log_p_lm + self.beta * num_words

        best = max(beams, key=score)
        return self._ids_to_text(best[0])

    # -----------------------------------------------------------------------
    # Provided — do NOT modify
    # -----------------------------------------------------------------------

    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        """
        Run the full decoding pipeline on a raw audio tensor.

        Args:
            audio_input (torch.Tensor): 1-D or 2-D audio waveform at 16 kHz.
            method (str): One of "greedy", "beam", "beam_lm", "beam_lm_rescore".

        Returns:
            str: Decoded transcript (lowercase).
        """
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values.squeeze(0)).logits[0]

        # Temperature scaling (Task 3): flatten/sharpen the distribution
        # before log_softmax.  T=1.0 is a no-op.  Your decoders must call
        # torch.log_softmax on the logits they receive — do not call it here.
        logits = logits / self.temperature

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose one of: 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'."
            )


# ---------------------------------------------------------------------------
# Quick debug helper — run this file directly to sanity-check your decoder
# on the provided examples/ clips before evaluating on the full test sets.
# ---------------------------------------------------------------------------


def test(decoder: Wav2Vec2Decoder, audio_path: str, reference: str) -> None:
    import jiwer

    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, f"Expected 16 kHz, got {sr} Hz for {audio_path}"

    print("=" * 60)
    print(f"REF : {reference}")

    for method in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        try:
            hyp = decoder.decode(audio_input, method=method)
        except NotImplementedError:
            print(f"  [{method}] not yet implemented")
            continue
        except ValueError as e:
            print(f"  [{method}] skipped ({e})")
            continue
        cer = jiwer.cer(reference, hyp)
        wer = jiwer.wer(reference, hyp)
        print(f"  [{method}] {hyp}")
        print(f"           WER={wer:.2%}  CER={cer:.2%}")


if __name__ == "__main__":
    # Reference transcripts are lowercase to match the evaluation manifests.
    # examples/ clips are for quick debugging only — use data/librispeech_test_other/
    # and data/earnings22_test/ for all reported metrics.
    test_samples = [
        (
            "examples/sample1.wav",
            "if you are generous here is a fitting opportunity for the exercise of your magnanimity if you are proud here am i your rival ready to acknowledge myself your debtor for an act of the most noble forbearance",
        ),
        (
            "examples/sample2.wav",
            "and if any of the other cops had private rackets of their own izzy was undoubtedly the man to find it out and use the information with a beat such as that even going halves and with all the graft to the upper brackets he'd still be able to make his pile in a matter of months",
        ),
        (
            "examples/sample3.wav",
            "guess a man gets used to anything hell maybe i can hire some bums to sit around and whoop it up when the ships come in and bill this as a real old martian den of sin",
        ),
        (
            "examples/sample4.wav",
            "it was a tune they had all heard hundreds of times so there was no difficulty in turning out a passable imitation of it to the improvised strains of i didn't want to do it the prisoner strode forth to freedom",
        ),
        (
            "examples/sample5.wav",
            "marguerite tired out with this long confession threw herself back on the sofa and to stifle a slight cough put up her handkerchief to her lips and from that to her eyes",
        ),
        ("examples/sample6.wav", "at this time all participants are in a listen only mode"),
        (
            "examples/sample7.wav",
            "the increase was mainly attributable to the net increase in the average size of our fleets",
        ),
        (
            "examples/sample8.wav",
            "operating surplus is a non cap financial measure which is defined as fully in our press release",
        ),
    ]

    decoder = Wav2Vec2Decoder()  # set lm_model_path for Tasks 4+

    for audio_path, reference in test_samples:
        test(decoder, audio_path, reference)
