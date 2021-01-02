import datetime
from pathlib import Path

import penman
from sacrebleu import corpus_bleu
import torch
from tqdm import tqdm
import smatch

from spring_amr.dataset import reverse_direction

def predict_amrs(
        loader, model, tokenizer, beam_size=1, tokens=None, restore_name_ops=False, return_all=False):

    shuffle_orig = loader.shuffle
    sort_orig = loader.sort

    loader.shuffle = False
    loader.sort = True

    total = len(loader.dataset)
    model.eval()
    model.amr_mode = True

    if tokens is None:
        ids = []
        tokens = []
        with tqdm(total=total) as bar:
            for x, y, extra in loader:
                ii = extra['ids']
                ids.extend(ii)
                with torch.no_grad():
                    out = model.generate(
                        **x,
                        max_length=1024,
                        decoder_start_token_id=0,
                        num_beams=beam_size,
                        num_return_sequences=beam_size)
                nseq = len(ii)
                for i1 in range(0, out.size(0), beam_size):
                    tokens_same_source = []
                    tokens.append(tokens_same_source)
                    for i2 in range(i1, i1+beam_size):
                        tokk = out[i2].tolist()
                        tokens_same_source.append(tokk)
                bar.update(nseq)
        # reorder
        tokens = [tokens[i] for i in ids]
        tokens = [t for tt in tokens for t in tt]

    graphs = []
    for i1 in range(0, len(tokens), beam_size):
        graphs_same_source = []
        graphs.append(graphs_same_source)
        for i2 in range(i1, i1+beam_size):
            tokk = tokens[i2]
            graph, status, (lin, backr) = tokenizer.decode_amr(tokk, restore_name_ops=restore_name_ops)
            graph.status = status
            graph.nodes = lin
            graph.backreferences = backr
            graph.tokens = tokk
            graphs_same_source.append(graph)
        graphs_same_source[:] = tuple(zip(*sorted(enumerate(graphs_same_source), key=lambda x: (x[1].status.value, x[0]))))[1]

    for gps, gg in zip(graphs, loader.dataset.graphs):
        for gp in gps:
            metadata = gg.metadata.copy()
            metadata['annotator'] = 'bart-amr'
            metadata['date'] = str(datetime.datetime.now())
            if 'save-date' in metadata:
                del metadata['save-date']
            gp.metadata = metadata

    loader.shuffle = shuffle_orig
    loader.sort = sort_orig

    if not return_all:
        graphs = [gg[0] for gg in graphs]

    return graphs

def predict_sentences(loader, model, tokenizer, beam_size=1, tokens=None, return_all=False):

    shuffle_orig = loader.shuffle
    sort_orig = loader.sort

    loader.shuffle = False
    loader.sort = True

    total = len(loader.dataset)
    model.eval()
    model.amr_mode = False
    
    if tokens is None:
        ids = []
        tokens = []
        with tqdm(total=total) as bar:
            for x, y, extra in loader:
                ids.extend(extra['ids'])
                x, y = reverse_direction(x, y)
                x['input_ids'] = x['input_ids'][:, :1024]
                x['attention_mask'] = x['attention_mask'][:, :1024]
                with torch.no_grad():
                    out = model.generate(
                        **x,
                        max_length=350,
                        decoder_start_token_id=0,
                        num_beams=beam_size,
                        num_return_sequences=beam_size)
                for i1 in range(0, len(out), beam_size):
                    tokens_same_source = []
                    tokens.append(tokens_same_source)
                    for i2 in range(i1, i1+beam_size):
                        tokk = out[i2]
                        tokk = [t for t in tokk.tolist() if t > 2]
                        tokens_same_source.append(tokk)
                bar.update(out.size(0) // beam_size)
        #reorder
        tokens = [tokens[i] for i in ids]

    sentences = []
    for tokens_same_source in tokens:
        if return_all:
            sentences.append([tokenizer.decode(tokk).strip() for tokk in tokens_same_source])
        else:
            sentences.append(tokenizer.decode(tokens_same_source[0]).strip())

    loader.shuffle = shuffle_orig
    loader.sort = sort_orig

    return sentences

def write_predictions(predictions_path, tokenizer, graphs):
    pieces = [penman.encode(g) for g in graphs]
    Path(predictions_path).write_text('\n\n'.join(pieces).replace(tokenizer.INIT, ''))
    return predictions_path

def compute_smatch(test_path, predictions_path):
    with Path(predictions_path).open() as p, Path(test_path).open() as g:
        score = next(smatch.score_amr_pairs(p, g))
    return score[2]

def compute_bleu(gold_sentences, pred_sentences):
    return corpus_bleu(pred_sentences, [gold_sentences])
