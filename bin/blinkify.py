import blink.main_dense as main_dense
from logging import getLogger
from penman import Triple, Graph
from spring_amr.evaluation import write_predictions
from spring_amr.tokenization_bart import AMRBartTokenizer
import json
from pathlib import Path
from spring_amr.IO import read_raw_amr_data
from spring_amr.entities import read_entities

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--blink-models-dir', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda',
        help="Device. 'cpu', 'cuda', 'cuda:<n>'.")
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()

    graphs = read_raw_amr_data(args.datasets)
    sentences = [g.metadata['snt'] for g in graphs]
    for_blink = []
    sample_id = 0

    for sent, (i, with_wikis, name_to_entity, name_to_ops) in zip(sentences, read_entities(sentences, graphs, just_tagged=not args.all)):
        for name, parent in name_to_entity.items():
            nt, wiki = with_wikis[parent]
            ops_triples = name_to_ops[name]
            ops_triples = sorted(ops_triples, key=lambda t: t[1])
            ops_triples = [t[2].strip('"') for t in ops_triples]
            string = ' '.join(ops_triples)
            found = string.lower() in sent.lower()
            if found:
                left = sent.lower().find(string.lower())
                right = left + len(string)

                sample = {
                    "id": sample_id,
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": sent[:left].strip().lower(),
                    "mention": string.lower(),
                    "context_right": sent[right:].strip().lower(),
                    "graph_n": i,
                    "triple_n": nt,
                }
                sample_id += 1
                for_blink.append(sample)

    main_dense.logger = logger = getLogger('BLINK')
    models_path = args.blink_models_dir # the path where you stored the BLINK models

    config = {
        "test_entities": None,
        "test_mentions": None,
        "interactive": False,
        "biencoder_model": models_path+"biencoder_wiki_large.bin",
        "biencoder_config": models_path+"biencoder_wiki_large.json",
        "entity_catalogue": models_path+"entity.jsonl",
        "entity_encoding": models_path+"all_entities_large.t7",
        "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
        "crossencoder_config": models_path+"crossencoder_wiki_large.json",
        "top_k": 10,
        "show_url": False,
        "fast": args.fast, # set this to be true if speed is a concern
        "output_path": models_path+"logs/", # logging directory
        "faiss_index": None,#"flat",
        "index_path": models_path+"faiss_flat_index.pkl",
    }

    args_blink = argparse.Namespace(**config)
    models = main_dense.load_models(args_blink, logger=logger)
    _, _, _, _, _, predictions, scores, = main_dense.run(args_blink, logger, *models, test_data=for_blink, device=args.device)

    for s, pp in zip(for_blink, predictions):
        pp = [p for p in pp if not p.startswith('List of')]
        p = f'"{pp[0]}"' if pp else '-'
        p = p.replace(' ', '_')
        graph_n = s['graph_n']
        triple_n = s['triple_n']
        triples = [g for g in graphs[graph_n].triples]
        n, rel, w = triples[triple_n]
        triples[triple_n] = Triple(n, rel, p)
        g = Graph(triples)
        g.metadata = graphs[graph_n].metadata
        graphs[graph_n] = g


    write_predictions(args.out, AMRBartTokenizer, graphs)
