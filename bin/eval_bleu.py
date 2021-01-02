import sys
import argparse
from typing import Iterable, Optional
import sacrebleu
import re


def argument_parser():

    parser = argparse.ArgumentParser(description='Preprocess AMR data')
    # Multiple input parameters
    parser.add_argument(
        "--in-tokens",
        help="input tokens",
        required=True,
        type=str
    )
    parser.add_argument(
        "--in-reference-tokens",
        help="refrence tokens to compute metric",
        type=str
    )
    args = parser.parse_args()

    return args


def tokenize_sentence(text, debug=False):
    text = re.sub(r"('ll|n't|'m|'s|'d|'re)", r" \1", text)
    text = re.sub(r"(\s+)", r" ", text)
    return text


def raw_corpus_bleu(hypothesis: Iterable[str], reference: Iterable[str],
                    offset: Optional[float] = 0.01) -> float:
    bleu = sacrebleu.corpus_bleu(hypothesis, reference, smooth_value=offset,
                                 force=True, use_effective_order=False,
                                 lowercase=True)
    return bleu.score


def raw_corpus_chrf(hypotheses: Iterable[str],
                    references: Iterable[str]) -> float:
    return sacrebleu.corpus_chrf(hypotheses, references,
                                 order=sacrebleu.CHRF_ORDER,
                                 beta=sacrebleu.CHRF_BETA,
                                 remove_whitespace=True)

def read_tokens(in_tokens_file):
    with open(in_tokens_file) as fid:
        lines = fid.readlines()
    return lines


if __name__ == '__main__':

    # Argument handlig
    args = argument_parser()

    # read files
    ref = read_tokens(args.in_reference_tokens)
    hyp = read_tokens(args.in_tokens)

    # Lower evaluation
    for i in range(len(ref)):
        ref[i] = ref[i].lower()

    # Lower case output
    for i in range(len(hyp)):
        if '<generate>' in hyp[i]:
            hyp[i] = hyp[i].split('<generate>')[-1]
        hyp[i] = tokenize_sentence(hyp[i].lower())

    # results

    bleu = raw_corpus_bleu(hyp, [ref])
    print('BLEU {:.2f}'.format(bleu))
    chrFpp = raw_corpus_chrf(hyp, ref).score * 100
    print('chrF++ {:.2f}'.format(chrFpp))