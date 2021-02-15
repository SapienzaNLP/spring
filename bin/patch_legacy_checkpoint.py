if __name__ == '__main__':

    from argparse import ArgumentParser
    import torch

    parser = ArgumentParser()
    parser.add_argument('legacy_checkpoint')
    parser.add_argument('patched_checkpoint')
    parser.parse_args()

    args = parser.parse_args()

    to_remove = []

    fixed = False
    w = torch.load(args.legacy_checkpoint, map_location='cpu')
    for name in w['model']:
        if 'backreferences' in name:
            fixed = True
            to_remove.append(name)
            print('Deleting parameters:', name)

    if not fixed:
        print('The checkpoint was fine as it was!')
    else:
        for name in to_remove:
            del w['model'][name]
        torch.save(w, args.patched_checkpoint)
