import pathlib
from copy import deepcopy
from typing import Dict
from difflib import ndiff
import argparse

from transformations.utils import UDLib
from transformations.utils.ucca.core import Passage
from transformations.utils.ucca.ioutil import read_files_and_dirs

#
# Transformations
#

# Scene-evoking nominals may correspond to clauses in translation.
# In order to avoid this discrepancy, we detect such nominals using
# UCCA parses and convert them to clauses.

def convert_nominal_predicates(
        ud_tree: UDLib.UDTree,
        ucca_parse: Passage,
        collapsed_participant_label: str = 'A'
):
    """
    Identifies nominal UD nodes that are analysed as P's in the corresponding
    UCCA passage and converts them to corresponding clauses:

    root          -> root
    nsubj         -> csubj
    nmod|compound -> acl
    obj|iobj      -> ccomp
    obl           -> advcl

    The resulting tree may be malformed UD as new clause nodes may have
    incorrect dependents. Consider applying collapse_participants and
    convert_amods to the result, in order to convert it to merely 'bad'
    UD (by collapsing participants to 'obl' or some other UD relation) or
    an intermediate UD-UCCA format where all participants are marked
    by a new tag 'A'.
    """

    relation_conversion_dict = {
        'nsubj': 'csubj',
        'nmod': 'acl',
        'compound': 'acl',
        'obj': 'ccomp',
        'iobj': 'ccomp',
        'obl': 'advcl'
    }
    adjectival_dependents_conversion_dict = {
        'amod': 'advmod',
        'acl': 'advcl'
    }
    nominal_dependents = ['compound', 'nmod']

    # Traverse the ucca_parse; find one-word tokens and check their categories.
    # If the category is P, look for the corresponding nominal token
    # in the UD parse. In an ideal world, we would also replace S's, but
    # they are not identified accurately enough.
    n_changes = 0
    ud_tree = deepcopy(ud_tree)
    tokens2node_keys = {
        ud_tree.nodes[node_key].FORM: node_key
        for node_key in ud_tree.keys
    }
    for node in ucca_parse.nodes.values():
        text = str(node).strip()
        if ' ' not in text:
            category = node.ftag if hasattr(node, "ftag") else node.tag
            if category == 'P':
                # Check if this token is analysed as a nominal in the
                # corresponding UD tree.
                ud_token_key = tokens2node_keys.get(text, None)
                if ud_token_key is not None:
                    ud_token = ud_tree.nodes[ud_token_key]
                    # Strip subcategories
                    deprel = ud_token.DEPREL.split(':')[0]
                    if deprel in relation_conversion_dict:
                        # print(f'{text:>12} : {category} -> {relation_conversion_dict[deprel]}')
                        # Change this token's deprel
                        n_changes += 1
                        replace_label(
                            ud_tree,
                            ud_token_key,
                            relation_conversion_dict[deprel])
                        # Change deprels of its dependents
                        for edge in ud_tree.graph[ud_token_key]:
                            # Strip subcategories
                            rel = edge.relation.split(':')[0]
                            if edge.directionality == 'down' and rel in nominal_dependents:
                                n_changes += 1
                                replace_label(
                                    ud_tree,
                                    edge.head,
                                    collapsed_participant_label)
                            elif edge.directionality == 'down' and rel in adjectival_dependents_conversion_dict:
                                n_changes += 1
                                replace_label(
                                    ud_tree,
                                    edge.head,
                                    adjectival_dependents_conversion_dict[rel])
    return ud_tree, n_changes


# After converting nominal predicates to clauses, we do not
# know which of their participants to promote to nsubj, obj, etc.,
# and so we collapse them. As a result, we get UD trees where
# some clauses have regular UD participants, while other have
# collapsed ones. This function harmonises this discrepancy.

def collapse_participants(
        ud_tree: UDLib.UDTree,
        collapsed_label: str = 'A'
):
    """
    Collapses nsubj, obj, iobj, and obl into A (when UCCA labels
    are adopted) or some other tag (e.g., obl) in place.
    """

    swap_labels(ud_tree, {
        relation: collapsed_label
        for relation in ['nsubj', 'obj', 'iobj', 'obl']
    })


#
# Helper functions
#


def get_top_level_ancestor(node):
    """
    Traverses the passage upwards and returns the node's
    ancestor that is immediately below the root.
    """
    # If node is already the root, return it.
    try:
        if not node.fparent:
            return node
        parent = node
        while parent.fparent.fparent:
            parent = parent.fparent
    except:
        return node
    return parent


def get_sent_id(tree: UDLib.UDTree):
    """
    Returns the sent_id of the tree if it is
    present; None otherwise.
    """
    for line in tree.id_lines:
        if line.startswith('# sent_id = '):
            return line.strip()[len('# sent_id = '):]
    else:
        return None


def get_tacred_sent_id(tree: UDLib.UDTree):
    """
    Returns the sent_id of the tree if it is
    present; None otherwise.
    """
    for line in tree.id_lines:
        if line.startswith('# id = '):
            return line.strip()[len('# id = '):]
    else:
        return None


def swap_labels(ud_tree: UDLib.UDTree, replacement_dict: Dict[str, str]):
    """
    Replaces labels in the tree according to the replacement dict in place.
    Labels' subcategories are ignored.
    """

    for node_key in ud_tree.keys:
        ud_tree.nodes[node_key].DEPREL = replacement_dict.get(
            ud_tree.nodes[node_key].DEPREL.split(':')[0],
            ud_tree.nodes[node_key].DEPREL)
        for i in range(len(ud_tree.graph[node_key])):
            ud_tree.graph[node_key][i].relation = replacement_dict.get(
                ud_tree.graph[node_key][i].relation.split(':')[0],
                ud_tree.graph[node_key][i].relation)


def replace_label(ud_tree: UDLib.UDTree, node_id: str, new_label: str):
    """
    Replaces the dependency label in the node and all edge goint out
    or coming into this node in place.
    """

    ud_tree.nodes[node_id].DEPREL = new_label
    for i, edge in enumerate(ud_tree.graph[node_id]):
        if edge.directionality == 'up':
            ud_tree.graph[node_id][i].relation = new_label
            # Replace the relation in the downward edge in the parent.
            parent_id = edge.head
            if parent_id != '0':
                for j, parent_edge in enumerate(ud_tree.graph[parent_id]):
                    if (
                            parent_edge.directionality == 'down' and
                            parent_edge.head == node_id
                    ):
                        ud_tree.graph[parent_id][j].relation = new_label
                        break


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Nominal Predicates Normalization Transformation")
    argparser.add_argument("-ud", "--ud-file", required=True)
    argparser.add_argument("-uc", "--ucca-dir", required=True)
    argparser.add_argument("-o", "--output-file", required=True)

    args = argparser.parse_args()

    # Iterate over parsed passages; find corresponding UD blocks.
    # Perform the replacements and show the diff of each affected block.
    collapse_participants_after_transform = True
    ucca_parses = read_files_and_dirs(args.ucca_dir)
    ud_trees = UDLib.conllu2trees(args.ud_file)
    ud_tree_dict = {get_sent_id(tree): tree for tree in ud_trees}
    transformed_ud_trees = []
    transformed_ud_trees_count = 0
    skipped = 0
    for ucca_parse in ucca_parses:
        sentence_id = ucca_parse.ID.rsplit('_', 1)[0]
        if sentence_id not in ud_tree_dict:
            skipped = skipped + 1
            continue
        print(sentence_id)
        ud_tree = ud_tree_dict[sentence_id]
        ud_tree_transformed, n_changes = convert_nominal_predicates(ud_tree, ucca_parse)
        if n_changes > 0:
            transformed_ud_trees_count = transformed_ud_trees_count + 1
            if collapse_participants_after_transform:
                collapse_participants(ud_tree_transformed)
            diff = ndiff(
                str(ud_tree).splitlines(keepends=True),
                str(ud_tree_transformed).splitlines(keepends=True))
        transformed_ud_trees.append(ud_tree_transformed)

    print(transformed_ud_trees_count)
    print(f'skipped:{skipped}')
    with open(args.output_file, 'w',
              encoding='utf-8') as output_file:
        for ud_tree in transformed_ud_trees:
            for line in str(ud_tree).splitlines(keepends=True):
                output_file.write(line)
            output_file.write('\n\n')
