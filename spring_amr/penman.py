from penman import load as load_, Graph, Triple
from penman import loads as loads_
from penman import encode as encode_
from penman.model import Model
from penman.models.noop import NoOpModel
from penman.models import amr

op_model = Model()
noop_model = NoOpModel()
amr_model = amr.model
DEFAULT = op_model

def _get_model(dereify):
    if dereify is None:
        return DEFAULT


    elif dereify:
        return op_model

    else:
        return noop_model

def _remove_wiki(graph):
    metadata = graph.metadata
    triples = []
    for t in graph.triples:
        v1, rel, v2 = t
        if rel == ':wiki':
            t = Triple(v1, rel, '+')
        triples.append(t)
    graph = Graph(triples)
    graph.metadata = metadata
    return graph

def load(source, dereify=None, remove_wiki=False):
    model = _get_model(dereify)
    out = load_(source=source, model=model)
    if remove_wiki:
        for i in range(len(out)):
            out[i] = _remove_wiki(out[i])
    return out

def loads(string, dereify=None, remove_wiki=False):
    model = _get_model(dereify)
    out = loads_(string=string, model=model)
    if remove_wiki:
        for i in range(len(out)):
            out[i] = _remove_wiki(out[i])
    return out

def encode(g, top=None, indent=-1, compact=False):
    model = amr_model
    return encode_(g=g, top=top, indent=indent, compact=compact, model=model)