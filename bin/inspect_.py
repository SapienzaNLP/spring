import torch
import penman
from spring_amr.utils import instantiate_model_and_tokenizer

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--beam-size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--penman-linearization', action='store_true',
        help="Predict using PENMAN linearization instead of ours.")
    parser.add_argument('--use-pointer-tokens', action='store_true')
    parser.add_argument('--restore-name-ops', action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device)
    model, tokenizer = instantiate_model_and_tokenizer(
        name='facebook/bart-large',
        checkpoint=args.checkpoint,
        dropout=0., attention_dropout=0.,
        penman_linearization=args.penman_linearization,
        use_pointer_tokens=args.use_pointer_tokens,
    )
    model.eval().to(device)

    while True:
        sentence = [input('Sentence to parse:\n')]
        x, extra = tokenizer.batch_encode_sentences(sentence, device)
        with torch.no_grad():
            out = model.generate(**x, max_length=1024, decoder_start_token_id=0, num_beams=args.beam_size)
        out = out[0].tolist()
        graph, status, (lin, backr) = tokenizer.decode_amr(out, restore_name_ops=args.restore_name_ops)
        print('-' * 5)
        print('Status:', status)
        print('-' * 5)
        print('Graph:')
        print(penman.encode(graph))
        print('-' * 5)
        print('Linearization:')
        print(lin)
        print('\n')
