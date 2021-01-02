from pathlib import Path

import penman
import torch

from spring_amr import ROOT
from spring_amr.evaluation import predict_amrs, compute_smatch, predict_sentences, compute_bleu
from spring_amr.penman import encode
from spring_amr.utils import instantiate_loader, instantiate_model_and_tokenizer

if __name__ == '__main__':

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Script to predict AMR graphs given sentences. LDC format as input.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--datasets', type=str, required=True, nargs='+',
                        help="Required. One or more glob patterns to use to load amr files.")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Required. Checkpoint to restore.")
    parser.add_argument('--model', type=str, default='facebook/bart-large',
                        help="Model config to use to load the model class.")
    parser.add_argument('--beam-size', type=int, default=1,
                        help="Beam size.")
    parser.add_argument('--batch-size', type=int, default=1000,
                        help="Batch size (as number of linearized graph tokens per batch).")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device. 'cpu', 'cuda', 'cuda:<n>'.")
    parser.add_argument('--pred-path', type=Path, default=ROOT / 'data/tmp/inf-pred-sentences.txt',
                        help="Where to write predictions.")
    parser.add_argument('--gold-path', type=Path, default=ROOT / 'data/tmp/inf-gold-sentences.txt',
                        help="Where to write the gold file.")
    parser.add_argument('--add-to-graph-file', action='store_true')
    parser.add_argument('--use-reverse-decoder', action='store_true')
    parser.add_argument('--deinvert', action='store_true')
    parser.add_argument('--penman-linearization', action='store_true',
        help="Predict using PENMAN linearization instead of ours.")
    parser.add_argument('--collapse-name-ops', action='store_true')
    parser.add_argument('--use-pointer-tokens', action='store_true')
    parser.add_argument('--raw-graph', action='store_true')
    parser.add_argument('--return-all', action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device)
    model, tokenizer = instantiate_model_and_tokenizer(
        args.model,
        dropout=0.,
        attention_dropout=0.,
        penman_linearization=args.penman_linearization,
        use_pointer_tokens=args.use_pointer_tokens,
        collapse_name_ops=args.collapse_name_ops,
        init_reverse=args.use_reverse_decoder,
        raw_graph=args.raw_graph,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model'])
    model.to(device)
    model.rev.amr_mode = False

    loader = instantiate_loader(
        args.datasets,
        tokenizer,
        batch_size=args.batch_size,
        evaluation=True, out='/tmp/a.txt',
        dereify=args.deinvert)
    loader.device = device

    pred_sentences = predict_sentences(loader, model.rev, tokenizer, beam_size=args.beam_size, return_all=args.return_all)
    if args.add_to_graph_file:
        graphs = loader.dataset.graphs
        for ss, g in zip(pred_sentences, graphs):
            if args.return_all:
                g.metadata['snt-pred'] = '\t\t'.join(ss)
            else:
                g.metadata['snt-pred'] = ss
        args.pred_path.write_text('\n\n'.join([encode(g) for g in graphs]))
    else:
        if args.return_all:
            pred_sentences = [s for ss in pred_sentences for s in ss]
        args.gold_path.write_text('\n'.join(loader.dataset.sentences))
        args.pred_path.write_text('\n'.join(pred_sentences))
        if not args.return_all:
            score = compute_bleu(loader.dataset.sentences, pred_sentences)
            print(f'BLEU: {score.score:.2f}')
