import argparse
import json
from copy import deepcopy
from typing import Dict, List, Tuple
from transformations.utils import UDLib

CONLLU_PATH = '../conllu'
OUT_PATH = '../conllu-processed'
SNACS_PATH = '../snacs-output/ewt'
ADVMOD_TAGS = {
    'Locus',
    'Time',
    'EndTime',
    'Goal',
    'Source',
    'Purpose',
    'Duration',
    'Circumstance',
    'ComparisonRef',
    'Manner',
    'Extent'
}


def get_annotations(path: str) -> List[Dict[str, List[str]]]:
    result = []
    with open(path, 'r', encoding='utf-8') as inp:
        for line in inp:
            result.append(json.loads(line))
    return result


def get_obliques_with_types(
        t: UDLib.UDTree,
        annotation: Dict[str, List[str]]) -> List[Tuple[str]]:
    assert len(t.keys) == len(annotation['tokens'])
    tag_dict = dict(zip(t.keys, annotation['tags']))
    result = []
    # DFS on the tree. Get the real root
    for edge in t.graph['0']:
        root = edge.head
    stack = [root]
    while stack:
        current_node = stack.pop()
        if t.nodes[current_node].DEPREL.split(':')[0] == 'obl':
            # Find the first case child and retrieve its annotation
            for edge in t.graph[current_node]:
                if edge.directionality == 'down':
                    child = edge.head
                    if t.nodes[child].DEPREL.split(':')[0] == 'case':
                        result.append((t.nodes[child].FORM + ' ' + t.nodes[current_node].FORM,
                                       tag_dict[child]))
                        break
            else:
                result.append((t.nodes[current_node].FORM, 'caseless'))
        for edge in t.graph[current_node]:
            if edge.directionality == 'down':
                stack.append(edge.head)
    return result


def transform_obl(t: UDLib.UDTree,
                  annotation: Dict[str, List[str]]) -> UDLib.UDTree:
    assert len(t.keys) == len(annotation['tokens'])
    tag_dict = dict(zip(t.keys, annotation['tags']))
    result = deepcopy(t)
    # Get the real root
    for edge in t.graph['0']:
        root = edge.head
    # DFS on the tree
    stack = [root]
    while stack:
        current_node = stack.pop()
        if t.nodes[current_node].DEPREL.split(':')[0] == 'obl':
            # Find the first _case_ child and retrieve its annotation
            for edge in t.graph[current_node]:
                if edge.directionality == 'down':
                    child = edge.head
                    if t.nodes[child].DEPREL.split(':')[0] == 'case':
                        tag = tag_dict[child].split('|')[0].split('.')[-1]
                        break
            else:
                tag = 'caseless'
            if tag in ADVMOD_TAGS:
                result.nodes[current_node].DEPREL = 'advmod'
            else:
                result.nodes[current_node].DEPREL = 'iobj'
        for edge in t.graph[current_node]:
            if edge.directionality == 'down':
                stack.append(edge.head)
    return result



if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Nominal Predicates Normalization Transformation")
    argparser.add_argument("-ud", "--ud-file", required=True)
    argparser.add_argument("-uc", "--snacs-file", required=True)
    argparser.add_argument("-o", "--output-file", required=True)

    args = argparser.parse_args()

    trees = UDLib.conllu2trees(args.ud_file)
    annotations = get_annotations(args.snacs_file)
    with open(args.output_file, 'w', encoding='utf-8'
    ) as out:
        for tree, annotation in zip(trees, annotations):
            print(transform_obl(tree, annotation), file=out)
            print('', file=out)