from pathlib import Path

import penman
import torch
from tqdm import tqdm

from spring_amr.penman import encode
from spring_amr.utils import instantiate_model_and_tokenizer

def read_file_in_batches(path, batch_size=1000, max_length=100):

    data = []
    idx = 0
    for line in Path(path).read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        n = len(line.split())
        if n > max_length:
            continue
        data.append((idx, line, n))
        idx += 1

    def _iterator(data):

        data = sorted(data, key=lambda x: x[2], reverse=True)

        maxn = 0
        batch = []

        for sample in data:
            idx, line, n = sample
            if n > batch_size:
                if batch:
                    yield batch
                    maxn = 0
                    batch = []
                yield [sample]
            else:
                curr_batch_size = maxn * len(batch)
                cand_batch_size = max(maxn, n) * (len(batch) + 1)

                if 0 < curr_batch_size <= batch_size and cand_batch_size > batch_size:
                    yield batch
                    maxn = 0
                    batch = []
                maxn = max(maxn, n)
                batch.append(sample)

        if batch:
            yield batch

    return _iterator(data), len(data)

if __name__ == '__main__':

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Script to predict AMR graphs given sentences. LDC format as input.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--texts', type=str, required=True, nargs='+',
        help="Required. One or more files containing \\n-separated sentences.")
    parser.add_argument('--checkpoint', type=str, required=True,
        help="Required. Checkpoint to restore.")
    parser.add_argument('--model', type=str, default='facebook/bart-large',
        help="Model config to use to load the model class.")
    parser.add_argument('--beam-size', type=int, default=1,
        help="Beam size.")
    parser.add_argument('--batch-size', type=int, default=1000,
        help="Batch size (as number of linearized graph tokens per batch).")
    parser.add_argument('--penman-linearization', action='store_true',
        help="Predict using PENMAN linearization instead of ours.")
    parser.add_argument('--use-pointer-tokens', action='store_true')
    parser.add_argument('--restore-name-ops', action='store_true')
    parser.add_argument('--device', type=str, default='cuda',
        help="Device. 'cpu', 'cuda', 'cuda:<n>'.")
    parser.add_argument('--only-ok', action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device)
    model, tokenizer = instantiate_model_and_tokenizer(
        args.model,
        dropout=0.,
        attention_dropout=0,
        penman_linearization=args.penman_linearization,
        use_pointer_tokens=args.use_pointer_tokens,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model'])
    model.to(device)
    model.eval()

    for path in tqdm(args.texts, desc='Files:'):

        iterator, nsent = read_file_in_batches(path, args.batch_size)

        with tqdm(desc=path, total=nsent) as bar:
            for batch in iterator:
                if not batch:
                    continue
                ids, sentences, _ = zip(*batch)
                x, _ = tokenizer.batch_encode_sentences(sentences, device=device)
                with torch.no_grad():
                    model.amr_mode = True
                    out = model.generate(**x, max_length=512, decoder_start_token_id=0, num_beams=args.beam_size)

                bgraphs = []
                for idx, sent, tokk in zip(ids, sentences, out):
                    graph, status, (lin, backr) = tokenizer.decode_amr(tokk.tolist(), restore_name_ops=args.restore_name_ops)
                    if args.only_ok and ('OK' not in str(status)):
                        continue
                    graph.metadata['status'] = str(status)
                    graph.metadata['source'] = path
                    graph.metadata['nsent'] = str(idx)
                    graph.metadata['snt'] = sent
                    bgraphs.append((idx, graph))

                for i, g in bgraphs:
                    print(encode(g))
                    print()

                # if bgraphs and args.reverse:
                #     bgraphs = [x[1] for x in bgraphs]
                #     x, _ = tokenizer.batch_encode_graphs(bgraphs, device)
                #     x = torch.cat([x['decoder_input_ids'], x['lm_labels'][:, -1:]], 1)
                #     att = torch.ones_like(x)
                #     att[att == tokenizer.pad_token_id] = 0
                #     x = {
                #         'input_ids': x,
                #         #'attention_mask': att,
                #     }
                #     with torch.no_grad():
                #         model.amr_mode = False
                #         out = model.generate(**x, max_length=1024, decoder_start_token_id=0, num_beams=args.beam_size)
                #
                #     for graph, tokk in zip(bgraphs, out):
                #         tokk = [t for t in tokk.tolist() if t > 2]
                #         graph.metadata['snt-pred'] = tokenizer.decode(tokk).strip()
                bar.update(len(sentences))

        exit(0)

        ids, graphs = zip(*sorted(results, key=lambda x:x[0]))

        for g in graphs:
            print(encode(g))
            print()
