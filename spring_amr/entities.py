from collections import defaultdict

def read_entities(sentences, graphs, just_tagged=True):

    for i, (s, g) in enumerate(zip(sentences, graphs)):

        with_wikis = {}
        name_to_entity = {}
        name_to_ops = defaultdict(list)

        for nt, t in enumerate(g.triples):
            n1, rel, n2 = t

            if n2 == '-' and just_tagged:
                continue

            if rel == ':wiki':
                with_wikis[n1] = (nt, n2)

        for t in g.triples:
            n1, rel, n2 = t
            if (n1 in with_wikis) and (rel == ':name'):
                name_to_entity[n2] = n1

        for nt, t in enumerate(g.triples):
            n1, rel, n2 = t
            if (n1 in name_to_entity) and rel.startswith(':op'):
                name_to_ops[n1].append(t)

        yield (i, with_wikis, name_to_entity, name_to_ops)