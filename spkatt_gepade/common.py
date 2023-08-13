import itertools
import json
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas


def to_idx(fname, sent_tok_str):
    x = sent_tok_str.split(':')
    return fname, int(x[0]), int(x[1])


def make_sentence_dataframe(sentence_objects, fname=''):
    df = []

    for sent in sentence_objects:
        for i, tok in enumerate(sent['Tokens']):
            df.append([fname, sent['SentenceId'], i, tok])

    df = pandas.DataFrame(df, columns=['filename', 'sentid', 'tokenid', 'token']).set_index(
        ['filename', 'sentid', 'tokenid'])
    df = df.sort_index()
    return df


def span_set_coverage(a, b):
    total = 0
    if len(b) == 0:
        return 0
    if len(a) == 0:
        return 0
    for b_ in b:
        total += max(len(a_ & b_) / len(a_) for a_ in a)

    return total / len(b)


def matching_precision_recall_f1(gold_annotations, pred_annotations, classes=None, aggregated=True):
    if classes is None:
        classes = ['Topic', 'Source', 'Evidence', 'Addr', 'Message', 'Medium', 'PTC', 'Cue']
    G = nx.Graph()

    for f, annotations in gold_annotations.items():
        for i, an in enumerate(annotations):
            if sum(len(an[k]) for k in classes) == 0:
                continue
            G.add_node((f, i, 'gold'), obj=an, denominator=sum(map(len, an.values())))
    for f, annotations in pred_annotations.items():
        for i, an in enumerate(annotations):
            if sum(len(an[k]) for k in classes) == 0:
                continue
            G.add_node((f, i, 'pred'), obj=an, denominator=sum(map(len, an.values())))

    G_precision = G.copy()
    G_recall = G.copy()
    for u, v in itertools.permutations(G.nodes, 2):
        if not (u[-1] == 'gold' and v[-1] == 'pred'):
            continue

        if u[0] != v[0]:
            G_precision.add_edge(u, v, weight=1, precision=0, nominator=0, denominator=G.nodes[v]['denominator'])
            G_recall.add_edge(u, v, weight=1, recall=0, nominator=0, denominator=G.nodes[u]['denominator'])
            continue

        obj_annot = G.nodes[u]['obj']
        obj_pred = G.nodes[v]['obj']

        tp, fp, fn = 0, 0, 0
        for schema in (classes if classes is not None else obj_annot.keys()):
            tp += len(set(obj_pred[schema]) & set(obj_annot[schema]))
            fp += len(set(obj_pred[schema]) - set(obj_annot[schema]))
            fn += len(set(obj_annot[schema]) - set(obj_pred[schema]))

        pr = tp / (tp + fp)
        rec = tp / (tp + fn)

        G_precision.add_edge(u, v, weight=1 - pr, precision=pr, nominator=tp, denominator=tp + fp)
        G_recall.add_edge(u, v, weight=1 - rec, recall=rec, nominator=tp, denominator=tp + fn)

    macro_precision, macro_recall = [], []
    G_precision_matched = nx.edge_subgraph(G_precision,
                                           nx.algorithms.bipartite.minimum_weight_full_matching(G_precision).items())
    G_recall_matched = nx.edge_subgraph(G_recall,
                                        nx.algorithms.bipartite.minimum_weight_full_matching(G_recall).items())
    for u in G.nodes:
        if u[-1] != 'pred': continue
        if u not in G_precision_matched.nodes:
            macro_precision.append([u, None] + [0, G.nodes[u]['denominator']])
        else:
            match = list(G_precision_matched.adj[u].keys())[0]
            macro_precision.append([u, match] + [G_precision.edges[u, match].get('nominator', 0),
                                                 G_precision.edges[u, match]['denominator']])
    for u in G.nodes:
        if u[-1] != 'gold': continue
        if u not in G_recall_matched.nodes:
            macro_precision.append([u, None] + [0, G.nodes[u]['denominator']])
        else:
            match = list(G_recall_matched.adj[u].keys())[0]
            macro_recall.append(
                [u, match] + [G_recall.edges[u, match].get('nominator', 0), G_recall.edges[u, match]['denominator']])

    if aggregated:
        precision = np.mean([k / n for _, _, k, n in macro_precision])
        recall = np.mean([k / n for _, _, k, n in macro_recall])
        return precision, recall, 2 / (1 / precision + 1 / recall)
    else:
        return macro_precision, macro_recall
