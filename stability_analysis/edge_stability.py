import argparse
import math
from typing import List, Union, Dict, Tuple

import pandas as pd
from conllu import parse_incr, parse, TokenList

from stability_analysis.config import Config
from stability_analysis.utilities import filter_multi_list, safe_div

AlignmentType = List[str]  # ["Aligned Edge", "Flipped Edge", "Unaligned Edge", "Unaligned Word"]


def _get_reversed_alignment_dict(dict: Dict[int, List[int]]):
    reversed_alignment_dict = {}
    for key in dict.keys():
        val_list = dict[key]
        for val in val_list:
            if val not in reversed_alignment_dict:
                reversed_alignment_dict[val] = []
            reversed_alignment_dict[val].append(key)

    return reversed_alignment_dict


def _find_by_id(id: int, token_list: TokenList):
    for token in token_list:
        if int(token["id"]) == id:
            return token
    raise KeyError(f'Can not find id {id} in token list with metadata {token_list.metadata}.')


def _get_edge_alignment(lang_token_id: int, lang_head_id: int, en_to_lang_alignment: Dict[int, List[int]],
                        en_conllu_data: TokenList) -> Tuple[AlignmentType, int, int]:
    lang_to_en_alignment = _get_reversed_alignment_dict(en_to_lang_alignment)

    if lang_token_id not in lang_to_en_alignment:
        return ("Non Content Word", None, None)

    if lang_head_id not in lang_to_en_alignment:
        return ("Non Content Word", None, None)

    if len(lang_to_en_alignment[lang_token_id]) > 1 or lang_to_en_alignment[lang_token_id][0] == -1:
        return ("Unaligned Words", None, None)

    if len(lang_to_en_alignment[lang_head_id]) > 1 or lang_to_en_alignment[lang_head_id][0] == -1:
        return ("Unaligned Words", None, None)

    aligned_en_token_id = lang_to_en_alignment[lang_token_id][0]
    aligned_en_head_id = lang_to_en_alignment[lang_head_id][0]

    assert aligned_en_token_id in en_to_lang_alignment
    assert aligned_en_head_id in en_to_lang_alignment
    if aligned_en_token_id not in en_to_lang_alignment:
        return ("Non-Content Word", None, None)
    if aligned_en_head_id not in en_to_lang_alignment:
        return ("Non-Content Word", None, None)

    assert en_to_lang_alignment[aligned_en_token_id][0] != -1
    if len(en_to_lang_alignment[aligned_en_token_id]) > 1 or en_to_lang_alignment[aligned_en_token_id][0] == -1:
        return ("Unaligned Words", None, None)

    assert en_to_lang_alignment[aligned_en_head_id][0] != -1
    if len(en_to_lang_alignment[aligned_en_head_id]) > 1 or en_to_lang_alignment[aligned_en_head_id][0] == -1:
        return ("Unaligned Words", None, None)

    aligned_lang_token_id = en_to_lang_alignment[aligned_en_token_id][0]
    aligned_lang_head_id = en_to_lang_alignment[aligned_en_head_id][0]

    assert lang_token_id == aligned_lang_token_id and lang_head_id == aligned_lang_head_id
    # "There is inconsistency is the word alignment dictionary")

    if _find_by_id(aligned_en_token_id, en_conllu_data)["head"] == aligned_en_head_id:
        return ("Aligned Edge", _find_by_id(aligned_en_token_id, en_conllu_data),
                _find_by_id(aligned_en_head_id, en_conllu_data))
    elif _find_by_id(aligned_en_head_id, en_conllu_data)["head"] == aligned_en_token_id:
        return ("Flipped Edge", _find_by_id(aligned_en_head_id, en_conllu_data),
                _find_by_id(aligned_en_token_id, en_conllu_data))
    else:
        return ("Unaligned Edge", None, None)


def _convert_str_keys_to_int(dict: Dict[str, List[str]]):
    new_dict = {}
    for key in dict:
        int_key = -1 if key == 'X' else int(key)
        new_dict[int_key] = []
        val_list = dict[key]
        for val in val_list:
            int_val = -1 if val == 'X' else int(val)
            new_dict[int_key].append(int_val)
    return new_dict


def get_aligned_edges_data_v2(lang: str, model_num: int,  with_pos:bool, use_supervised_parse: bool = False):
    pd.set_option('display.max_columns', 1000)

    import sqlite3
    import json

    conn = sqlite3.connect("./data/pud_current.db")
    """res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for name in res:
        print(name[0])"""

    df = pd.read_sql_query(f'SELECT * FROM "en-{lang}"', conn)

    zero_shot_prediction_dict = {}
    lang2 = lang if lang != "jp" else "ja"
    if use_supervised_parse:
        zero_shot_prediction_path = f'{Config.pud_zero_shot_prediction_dir}/model_{model_num}/{lang2}_pud-ud-test.conllu.prediction'
    else:
        zero_shot_prediction_path = f'{Config.pud_supervised_prediction_dir}/model_{model_num}/{lang2}_pud-ud-test.conllu.prediction'
    with open(zero_shot_prediction_path, 'r', encoding="utf-8") as prediction_file:
        zero_shot_prediction = list(parse_incr(prediction_file))

        for i, sentence in enumerate(zero_shot_prediction):
            sent_id = sentence.metadata["sent_id"]
            assert sent_id not in zero_shot_prediction_dict
            zero_shot_prediction_dict[sent_id] = zero_shot_prediction[i]

    # --- Edge Type: ----
    # Aligned: Edge that have a 1-1 aligned edge, including same head->dependency direction.
    # Flipped: Edges that have a 1-1 aligned edge, but the head->dependency relationship is flipped to dependency->head
    # Unaligned: Edges that don't have an aligned edge.
    edge_type: List[AlignmentType] = []
    # Head Distance: (token_id)-(head_id)
    predicted_edge_distance: List[int] = []
    actual_edge_distance: List[int] = []
    # If edge is of type 'Unaligned' the value is None
    projected_edge_distance: List[Union[int, None]] = []
    predicted_label: List[str] = []
    actual_label: List[str] = []
    # If edge is of type 'Unaligned' the value is None
    projected_label: List[Union[str, None]] = []

    for sentence in df.index:
        if sentence == 507:
            continue

        document_id = df.loc[sentence]['document_id']
        sentence_id = df.loc[sentence]['sentence_id']

        # the [0] indexing is because the json is constructed as as list of length 1
        en_conllu_data = list(filter(lambda tok: isinstance(tok["id"], int), parse(df.loc[sentence]['en'])[0]))
        lang_conllu_data = list(filter(lambda tok: isinstance(tok["id"], int), parse(df.loc[sentence]["ru"])[0]))
        lang_predicted_conllu_data = list(
            filter(lambda tok: isinstance(tok["id"], int), zero_shot_prediction_dict[sentence_id]))

        en_to_lang_alignment = _convert_str_keys_to_int(json.loads(df.loc[sentence]['alignment']))
        for lang_token, lang_predicted_token in zip(lang_conllu_data, lang_predicted_conllu_data):
            lang_token_id = lang_token["id"]
            lang_head_id = lang_token["head"]
            assert lang_token["id"] == lang_predicted_token["id"]

            (alignment_type, aligned_en_token, aligned_en_head) = \
                _get_edge_alignment(lang_token_id, lang_head_id, en_to_lang_alignment, en_conllu_data)

            edge_type.append(alignment_type)
            predicted_edge_distance.append(lang_predicted_token["head"] - lang_predicted_token["id"])
            actual_edge_distance.append(lang_token["head"] - lang_token["id"])
            predicted_label.append(lang_predicted_token["deprel"].split(":")[0])
            actual_label.append(lang_token["deprel"].split(":")[0])
            projected_edge_distance.append(aligned_en_head["id"] - aligned_en_token["id"]
                                           if aligned_en_head is not None and aligned_en_token is not None else None)
            projected_label.append(aligned_en_token["deprel"].split(":")[0]
                                   if aligned_en_head is not None and aligned_en_token is not None else None)

    return edge_type, predicted_edge_distance, actual_edge_distance, projected_edge_distance, \
           predicted_label, actual_label, projected_label


def experiment_1_base(lang: str, model_num: int, with_pos: bool, use_supervised_parse: bool = False, label_to_filter_by: str = None) -> pd.DataFrame:
    round_count = 100
    pd.set_option('display.max_columns', 1000)
    edge_type, predicted_edge_distance, actual_edge_distance, projected_edge_distance, \
    predicted_label, actual_label, projected_label = get_aligned_edges_data_v2(lang, model_num, with_pos,
                                                                               use_supervised_parse)

    if label_to_filter_by is not None:
        actual_label, [edge_type, predicted_edge_distance, actual_edge_distance, projected_edge_distance,
                       predicted_label, projected_label] = filter_multi_list(actual_label,
                                                                             [edge_type,
                                                                              predicted_edge_distance,
                                                                              actual_edge_distance,
                                                                              projected_edge_distance,
                                                                              predicted_label,
                                                                              projected_label],
                                                                             label_to_filter_by)

    aligned_edges_unlabeled_correct = 0
    aligned_edges_count = len(list(filter(lambda x: x == "Aligned Edge", edge_type)))
    flipped_edges_unlabeled_correct = 0
    flipped_edges_unlabeled_flipped_pred = 0
    flipped_edges_unlabeled_flipped_dir_pred = 0
    flipped_edges_count = len(list(filter(lambda x: x == "Flipped Edge", edge_type)))
    unaligned_edges_unlabeled_correct = 0
    unaligned_edges_count = len(list(filter(lambda x: x == "Unaligned Edge", edge_type)))
    unaligned_words_unlabeled_correct = 0
    unaligned_edges_unlabeled_proj_pred = 0
    unaligned_words_count = len(list(filter(lambda x: x == "Unaligned Words", edge_type)))
    non_content_word_edges_unlabeled_correct = 0
    non_content_word_edges_count = len(list(filter(lambda x: x == "Non Content Word", edge_type)))
    for type, predicted, actual, projected in zip(edge_type, predicted_edge_distance, actual_edge_distance,
                                                  projected_edge_distance):
        if type == "Aligned Edge" and predicted == actual:
            aligned_edges_unlabeled_correct = aligned_edges_unlabeled_correct + 1
        if type == "Flipped Edge" and predicted == actual:
            flipped_edges_unlabeled_correct = flipped_edges_unlabeled_correct + 1
        # need to think how to do this
        if type == "Flipped Edge" and predicted == -1 * actual:
            flipped_edges_unlabeled_flipped_pred = flipped_edges_unlabeled_flipped_pred + 1
        if type == "Flipped Edge" and math.copysign(predicted, -1 * actual):
            flipped_edges_unlabeled_flipped_dir_pred = flipped_edges_unlabeled_flipped_dir_pred + 1
        if type == "Unaligned Edge" and predicted == actual:
            unaligned_edges_unlabeled_correct = unaligned_edges_unlabeled_correct + 1
        if type == "Unaligned Edge" and predicted == projected:
            # TODO: Think how to cacl this. I want predicted == the rpojected of the algned words in the target
            # Probably need to add to the function another value
            unaligned_edges_unlabeled_proj_pred = unaligned_edges_unlabeled_proj_pred + 1
        if type == "Unaligned Words" and predicted == actual:
            unaligned_words_unlabeled_correct = unaligned_words_unlabeled_correct + 1
        if type == "Non Content Word" and predicted == actual:
            non_content_word_edges_unlabeled_correct = non_content_word_edges_unlabeled_correct + 1

    columns_1 = [lang]
    column_2 = ["Count", "Percentage", "ZS UAS"]
    columns = pd.MultiIndex.from_product([columns_1, column_2])
    index = ["Aligned Edges", "Flipped Edges", "Unaligned Edges", "Unaligned Words", "Non Content Word"]
    if label_to_filter_by is not None:
        index = pd.MultiIndex.from_product([[label_to_filter_by], index])
    data = [[aligned_edges_count, round(safe_div(aligned_edges_count, len(edge_type)), round_count),
             round(safe_div(aligned_edges_unlabeled_correct, aligned_edges_count), round_count)],
            [flipped_edges_count, round(safe_div(flipped_edges_count, len(edge_type)), round_count),
             round(safe_div(flipped_edges_unlabeled_correct, flipped_edges_count), round_count)],
            [unaligned_edges_count, round(safe_div(unaligned_edges_count, len(edge_type)), round_count),
             round(safe_div(unaligned_edges_unlabeled_correct, unaligned_edges_count), round_count)],
            [unaligned_words_count, round(safe_div(unaligned_words_count, len(edge_type)), round_count),
             round(safe_div(unaligned_words_unlabeled_correct, unaligned_words_count), round_count)],
            [non_content_word_edges_count, round(safe_div(non_content_word_edges_count, len(edge_type)), round_count),
             round(safe_div(non_content_word_edges_unlabeled_correct, non_content_word_edges_count), round_count)]
            ]
    df = pd.DataFrame(data=data, index=index, columns=columns)

    return df


def experiment_1_base_analysis(lang: str, model_num: int, with_pos: bool,
                               label_to_filter_by: str = None) -> pd.DataFrame:
    round_count = 100
    pd.set_option('display.max_columns', 1000)
    edge_type, predicted_edge_distance, actual_edge_distance, projected_edge_distance, \
    predicted_label, actual_label, projected_label = get_aligned_edges_data_v2(lang, model_num, with_pos)

    if label_to_filter_by is not None:
        actual_label, [edge_type, predicted_edge_distance, actual_edge_distance, projected_edge_distance,
                       predicted_label, projected_label] = filter_multi_list(actual_label,
                                                                             [edge_type,
                                                                              predicted_edge_distance,
                                                                              actual_edge_distance,
                                                                              projected_edge_distance,
                                                                              predicted_label,
                                                                              projected_label],
                                                                             label_to_filter_by)

    aligned_edges_unlabeled_correct = 0
    aligned_edges_count = len(list(filter(lambda x: x == "Aligned Edge", edge_type)))
    flipped_edges_unlabeled_correct = 0
    flipped_edges_unlabeled_flipped_pred = 0
    flipped_edges_unlabeled_flipped_dir_pred = 0
    flipped_edges_count = len(list(filter(lambda x: x == "Flipped Edge", edge_type)))
    unaligned_edges_unlabeled_correct = 0
    unaligned_edges_count = len(list(filter(lambda x: x == "Unaligned Edge", edge_type)))
    unaligned_words_unlabeled_correct = 0
    unaligned_edges_unlabeled_proj_pred = 0
    unaligned_words_count = len(list(filter(lambda x: x == "Unaligned Words", edge_type)))
    non_content_word_edges_unlabeled_correct = 0
    non_content_word_edges_count = len(list(filter(lambda x: x == "Non Content Word", edge_type)))
    for type, predicted, actual, projected in zip(edge_type, predicted_edge_distance, actual_edge_distance,
                                                  projected_edge_distance):
        if type == "Aligned Edge" and predicted == actual:
            aligned_edges_unlabeled_correct = aligned_edges_unlabeled_correct + 1
        if type == "Flipped Edge" and predicted == actual:
            flipped_edges_unlabeled_correct = flipped_edges_unlabeled_correct + 1
        if type == "Flipped Edge" and predicted == -1 * actual:
            flipped_edges_unlabeled_flipped_pred = flipped_edges_unlabeled_flipped_pred + 1
        if type == "Flipped Edge" and math.copysign(1, predicted) == math.copysign(1, -1 * actual):
            flipped_edges_unlabeled_flipped_dir_pred = flipped_edges_unlabeled_flipped_dir_pred + 1
        if type == "Unaligned Edge" and predicted == actual:
            unaligned_edges_unlabeled_correct = unaligned_edges_unlabeled_correct + 1
        if type == "Unaligned Edge" and predicted == projected:
            # TODO: Think how to cacl this. I want predicted == the rpojected of the algned words in the target
            # Probably need to add to the function another value
            unaligned_edges_unlabeled_proj_pred = unaligned_edges_unlabeled_proj_pred + 1
        if type == "Unaligned Words" and predicted == actual:
            unaligned_words_unlabeled_correct = unaligned_words_unlabeled_correct + 1
        if type == "Non Content Word" and predicted == actual:
            non_content_word_edges_unlabeled_correct = non_content_word_edges_unlabeled_correct + 1

    flipped_edges_predicted_as_exact_flipped_percentage = safe_div(flipped_edges_unlabeled_flipped_pred,
                                                                   flipped_edges_count)
    flipped_edges_predicted_opposite_direction_percentage = safe_div(flipped_edges_unlabeled_flipped_dir_pred,
                                                                     flipped_edges_count)
    columns_1 = [lang]
    column_2 = ["Count", "Percentage", "ZS UAS"]
    columns = pd.MultiIndex.from_product([columns_1, column_2])
    index = ["Aligned Edges", "Flipped Edges", "Unaligned Edges", "Unaligned Words", "Non Content Word"]
    if label_to_filter_by is not None:
        index = pd.MultiIndex.from_product([[label_to_filter_by], index])
    data = [[aligned_edges_count, round(safe_div(aligned_edges_count, len(edge_type)), round_count),
             round(safe_div(aligned_edges_unlabeled_correct, aligned_edges_count), round_count)],
            [flipped_edges_count, round(safe_div(flipped_edges_count, len(edge_type)), round_count),
             round(safe_div(flipped_edges_unlabeled_correct, flipped_edges_count), round_count)],
            [unaligned_edges_count, round(safe_div(unaligned_edges_count, len(edge_type)), round_count),
             round(safe_div(unaligned_edges_unlabeled_correct, unaligned_edges_count), round_count)],
            [unaligned_words_count, round(safe_div(unaligned_words_count, len(edge_type)), round_count),
             round(safe_div(unaligned_words_unlabeled_correct, unaligned_words_count), round_count)],
            [non_content_word_edges_count, round(safe_div(non_content_word_edges_count, len(edge_type)), round_count),
             round(safe_div(non_content_word_edges_unlabeled_correct, non_content_word_edges_count), round_count)]
            ]
    df = pd.DataFrame(data=data, index=index, columns=columns)

    return df


def experiment_1_base_labeled_score(lang: str, model_num: int, with_pos: bool, use_supervised_parse: bool,
                                    label_to_filter_by: str = None) -> pd.DataFrame:
    round_count = 100
    pd.set_option('display.max_columns', 1000)
    edge_type, predicted_edge_distance, actual_edge_distance, projected_edge_distance, \
    predicted_label, actual_label, projected_label = get_aligned_edges_data_v2(lang, model_num, with_pos, use_supervised_parse)

    for label in actual_label:
        assert len(label.split(":")) == 1
    for label in predicted_label:
        assert len(label.split(":")) == 1
    for label in projected_label:
        assert label is None or len(label.split(":")) == 1

    if label_to_filter_by is not None:
        actual_label, [edge_type, predicted_edge_distance, actual_edge_distance, projected_edge_distance,
                       predicted_label, projected_label] = filter_multi_list(actual_label,
                                                                             [edge_type,
                                                                              predicted_edge_distance,
                                                                              actual_edge_distance,
                                                                              projected_edge_distance,
                                                                              predicted_label,
                                                                              projected_label],
                                                                             label_to_filter_by)
    fully_aligned_edges_count = 0
    fully_aligned_edges_unlabeled_correct = 0
    fully_aligned_edges_labeled_correct = 0
    partially_aligned_edges_count = 0
    partially_aligned_edges_unlabeled_correct = 0
    partially_aligned_edges_labeled_correct = 0
    aligned_edges_unlabeled_correct = 0
    aligned_edges_labeled_correct = 0
    aligned_edges_count = len(list(filter(lambda x: x == "Aligned Edge", edge_type)))
    flipped_edges_unlabeled_correct = 0
    flipped_edges_labeled_correct = 0
    flipped_edges_count = len(list(filter(lambda x: x == "Flipped Edge", edge_type)))
    unaligned_edges_unlabeled_correct = 0
    unaligned_edges_labeled_correct = 0
    unaligned_edges_count = len(list(filter(lambda x: x == "Unaligned Edge", edge_type)))
    unaligned_words_unlabeled_correct = 0
    unaligned_words_labeled_correct = 0
    unaligned_words_count = len(list(filter(lambda x: x == "Unaligned Words", edge_type)))
    non_content_word_edges_unlabeled_correct = 0
    non_content_word_edges_labeled_correct = 0
    non_content_word_edges_count = len(list(filter(lambda x: x == "Non Content Word", edge_type)))
    for type, predicted_e, actual_e, predicted_l, actual_l, proj_l in \
            zip(edge_type, predicted_edge_distance, actual_edge_distance, predicted_label, actual_label, projected_label):
        if type == "Aligned Edge" and actual_l == proj_l:
            fully_aligned_edges_count = fully_aligned_edges_count + 1
            if predicted_e == actual_e and predicted_l == actual_l:
                fully_aligned_edges_labeled_correct = fully_aligned_edges_labeled_correct + 1
            if predicted_e == actual_e:
                fully_aligned_edges_unlabeled_correct = fully_aligned_edges_unlabeled_correct + 1
        if type == "Aligned Edge" and actual_l != proj_l:
            partially_aligned_edges_count = partially_aligned_edges_count + 1
            if predicted_e == actual_e and predicted_l == actual_l:
                partially_aligned_edges_labeled_correct = partially_aligned_edges_labeled_correct + 1
            if predicted_e == actual_e:
                partially_aligned_edges_unlabeled_correct = partially_aligned_edges_unlabeled_correct + 1
        if type == "Aligned Edge" and predicted_e == actual_e and predicted_l == actual_l:
            aligned_edges_labeled_correct = aligned_edges_labeled_correct + 1
        if type == "Flipped Edge" and predicted_e == actual_e and predicted_l == actual_l:
            flipped_edges_labeled_correct = flipped_edges_labeled_correct + 1
        if type == "Unaligned Edge" and predicted_e == actual_e and predicted_l == actual_l:
            unaligned_edges_labeled_correct = unaligned_edges_labeled_correct + 1
        if type == "Unaligned Words" and predicted_e == actual_e and predicted_l == actual_l:
            unaligned_words_labeled_correct = unaligned_words_labeled_correct + 1
        if type == "Non Content Word" and predicted_e == actual_e and predicted_l == actual_l:
            non_content_word_edges_labeled_correct = non_content_word_edges_labeled_correct + 1

        if type == "Aligned Edge" and predicted_e == actual_e:
            aligned_edges_unlabeled_correct = aligned_edges_unlabeled_correct + 1
        if type == "Flipped Edge" and predicted_e == actual_e:
            flipped_edges_unlabeled_correct = flipped_edges_unlabeled_correct + 1
        if type == "Unaligned Edge" and predicted_e == actual_e:
            unaligned_edges_unlabeled_correct = unaligned_edges_unlabeled_correct + 1
        if type == "Unaligned Words" and predicted_e == actual_e:
            unaligned_words_unlabeled_correct = unaligned_words_unlabeled_correct + 1
        if type == "Non Content Word" and predicted_e == actual_e:
            non_content_word_edges_unlabeled_correct = non_content_word_edges_unlabeled_correct + 1

    columns_1 = [lang]
    column_2 = ["Count", "Percentage", "ZS UAS", "ZS LAS"]
    columns = pd.MultiIndex.from_product([columns_1, column_2])
    index = ["Fully Aligned Edges", "Partially Aligned Edges", "Aligned Edges", "Flipped Edges", "Unaligned Edges", "Unaligned Words", "Non Content Word"]
    if label_to_filter_by is not None:
        index = pd.MultiIndex.from_product([[label_to_filter_by], index])
    data = [[fully_aligned_edges_count,
             round(safe_div(fully_aligned_edges_count, len(edge_type)), round_count)*100,
             round(safe_div(fully_aligned_edges_unlabeled_correct, fully_aligned_edges_count), round_count)*100,
             round(safe_div(fully_aligned_edges_labeled_correct, fully_aligned_edges_count), round_count)*100],
            [partially_aligned_edges_count,
             round(safe_div(partially_aligned_edges_count, len(edge_type)), round_count) * 100,
             round(safe_div(partially_aligned_edges_unlabeled_correct, partially_aligned_edges_count), round_count) * 100,
             round(safe_div(partially_aligned_edges_labeled_correct, partially_aligned_edges_count), round_count) * 100],
            [aligned_edges_count,
             round(safe_div(aligned_edges_count, len(edge_type)), round_count)*100,
             round(safe_div(aligned_edges_unlabeled_correct, aligned_edges_count), round_count)*100,
             round(safe_div(aligned_edges_labeled_correct, aligned_edges_count), round_count)*100],
            [flipped_edges_count,
             round(safe_div(flipped_edges_count, len(edge_type)), round_count)*100,
             round(safe_div(flipped_edges_unlabeled_correct, flipped_edges_count), round_count)*100,
             round(safe_div(flipped_edges_labeled_correct, flipped_edges_count), round_count)*100],
            [unaligned_edges_count,
             round(safe_div(unaligned_edges_count, len(edge_type)), round_count)*100,
             round(safe_div(unaligned_edges_unlabeled_correct, unaligned_edges_count), round_count)*100,
             round(safe_div(unaligned_edges_labeled_correct, unaligned_edges_count), round_count)*100],
            [unaligned_words_count,
             round(safe_div(unaligned_words_count, len(edge_type)), round_count)*100,
             round(safe_div(unaligned_words_unlabeled_correct, unaligned_words_count), round_count)*100,
             round(safe_div(unaligned_words_labeled_correct, unaligned_words_count), round_count)*100],
            [non_content_word_edges_count,
             round(safe_div(non_content_word_edges_count, len(edge_type)), round_count)*100,
             round(safe_div(non_content_word_edges_unlabeled_correct, non_content_word_edges_count), round_count)*100,
             round(safe_div(non_content_word_edges_labeled_correct, non_content_word_edges_count), round_count)*100]
            ]
    df = pd.DataFrame(data=data, index=index, columns=columns)

    return df


def experiment_1(use_supervised_parse: bool, language: str):
    with_pos = None  # Param removed
    df_list = []
    std_df_list = []
    for lang in [language]:
        # print(f'Evaluating {lang}')
        lang_df_list = []
        for i in range(10):
            # print(f'Evaluating model {i + 1}...')
            df = experiment_1_base_labeled_score(lang, i + 1, with_pos, use_supervised_parse)

            lang_df_list.append(df)
        df = round(safe_div(sum(lang_df_list), len(lang_df_list)), 3)
        df_list.append(df)

        las_std_df = pd.concat([d[lang]['ZS LAS'] for d in lang_df_list], axis=1).std(axis=1)
        uas_std_df = pd.concat([d[lang]['ZS UAS'] for d in lang_df_list], axis=1).std(axis=1)
        data = uas_std_df.rename("ZS UAS STD").to_frame().join(las_std_df.rename("ZS LAS STD"))

        columns_1 = [lang]
        column_2 = data.columns
        columns = pd.MultiIndex.from_product([columns_1, column_2])
        assert all(las_std_df.index == uas_std_df.index)
        index = las_std_df.index
        data = uas_std_df.rename("ZS UAS STD").to_frame().join(las_std_df.rename("ZS LAS STD"))
        std_df = pd.DataFrame(data=data.values, index=index, columns=columns)

        std_df_list.append(std_df)

    # print("Done!")
    df = pd.concat(df_list, axis=1)
    std_df = pd.concat(std_df_list, axis=1)
    print(df)
    print(std_df)


def experiment_3_base_v2(lang: str, model_num: int, with_pos: bool, is_projection_pool: bool):
    round_count = 100
    pd.set_option('display.max_columns', 1000)
    edge_type, predicted_edge_distance, actual_edge_distance, projected_edge_distance, \
    predicted_label, actual_label, projected_label = get_aligned_edges_data_v2(lang, model_num, with_pos)

    for label in actual_label:
        assert len(label.split(":")) == 1
    for label in predicted_label:
        assert len(label.split(":")) == 1
    for label in projected_label:
        if label is None:
            continue
        assert len(label.split(":")) == 1

    total = 0
    unlabeled_correct = 0
    label_and_edge_correct = 0
    correct_and_predicted_as_projection = 0
    only_label_correct = 0
    only_label_correct_and_predicted_as_projection = 0
    only_edge_correct = 0
    only_edge_correct_and_predicted_as_projection = 0
    incorrect = 0
    incorrect_and_predicted_as_projection = 0
    for e_type, a_distance, p_distance, a_label, p_label, proj_label in \
            zip(edge_type, actual_edge_distance, predicted_edge_distance,
                actual_label, predicted_label, projected_label):
        if e_type not in ["Aligned Edge"]:  # , "Flipped Edge"]:
            continue

        assert proj_label is not None

        if is_projection_pool and a_label != proj_label:
            continue
        if not is_projection_pool and a_label == proj_label:
            continue

        total = total + 1
        if a_distance == p_distance:
            unlabeled_correct = unlabeled_correct + 1

        if a_distance == p_distance and a_label == p_label:
            label_and_edge_correct = label_and_edge_correct + 1
            if proj_label == p_label:
                correct_and_predicted_as_projection = correct_and_predicted_as_projection + 1
        else:
            incorrect = incorrect + 1
            if proj_label == p_label:
                incorrect_and_predicted_as_projection = incorrect_and_predicted_as_projection + 1

            if a_label == p_label:
                only_label_correct = only_label_correct + 1
                if proj_label == p_label:
                    only_label_correct_and_predicted_as_projection = only_label_correct_and_predicted_as_projection + 1
            if a_distance == p_distance:
                only_edge_correct = only_edge_correct + 1
                if proj_label == p_label:
                    only_edge_correct_and_predicted_as_projection = only_edge_correct_and_predicted_as_projection + 1

    columns = ["Count", "ZS LAS", "ZS UAS",
               "Correct and Predicted as Projection", "Incorrect", "Incorrect and Predicted as Projection",
               "ZS Only Label Recall", "Only Label Correct and Predicted as Projection",
               "only_edge_correct", "only_edge_correct_and_predicted_as_projection",
               "only_edge_correct_and_predicted_as_projection_out_of_error",
               "only_edge_correct_and_predicted_as_projection_out_of_total"]
    index_1 = [lang]
    index_2 = ["Aligned Edges"] if is_projection_pool else ["Partially Aligned Edges"]
    index = pd.MultiIndex.from_product([index_1, index_2])
    # if label_to_filter_by is not None:
    #    index = pd.MultiIndex.from_product([[label_to_filter_by], index])
    data = [[total,
             round(safe_div(label_and_edge_correct, total), round_count)*100,
             round(safe_div(unlabeled_correct, total), round_count)*100,
             round(correct_and_predicted_as_projection, round_count),
             round(incorrect, round_count),
             round(incorrect_and_predicted_as_projection, round_count),
             round(only_label_correct, round_count), round(only_label_correct_and_predicted_as_projection, round_count),
             round(only_edge_correct, round_count), round(only_edge_correct_and_predicted_as_projection, round_count),
             round(safe_div(only_edge_correct_and_predicted_as_projection,incorrect), round_count)*100,
             round(safe_div(only_edge_correct_and_predicted_as_projection, total), round_count)*100
             ]]
    df = pd.DataFrame(data=data, index=index, columns=columns)

    return df


def experiment_3(is_projection_pool: bool, language: str):
    with_pos = None  # Param removed
    df_list = []
    std_dict = {}
    for lang in [language]:
        # print(f'Evaluating {lang}...')
        lang_df_list = []
        for i in range(10):
            # print(f'Evaluating model {i + 1}...')
            df = experiment_3_base_v2(lang, i + 1, with_pos, is_projection_pool)
            lang_df_list.append(df)
        df = round(safe_div(sum(lang_df_list), len(lang_df_list)), 3)
        df_list.append(df)
        std_df = pd.concat(lang_df_list, axis=0).std(axis=0)
        std_dict[lang] = std_df

    # print("Done!")
    df = pd.concat(df_list, axis=0)
    print(df)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Evaluating method for Universal Dependencies")
    argparser.add_argument("-l", "--language", required=True)
    args = argparser.parse_args()

    pd.set_option('display.max_columns', 1000)
    print("Edge Stability Categories Stats - Supervised Parse")
    print("--------------------------------------------------")
    experiment_1(use_supervised_parse=True, language=args.language)
    print("Edge Stability Categories Stats - Zero-Shot Parse")
    print("--------------------------------------------------")
    experiment_1(use_supervised_parse=False, language=args.language)
    print("Aligned Edges Stats - Zero-Shot Parse")
    print("-------------------------------------")
    experiment_3(is_projection_pool=True, language=args.language)
    print("Partially Aligned Edges Stats - Zero-Shot Parse")
    print("-----------------------------------------------")
    experiment_3(is_projection_pool=False, language=args.language)

    exit()
