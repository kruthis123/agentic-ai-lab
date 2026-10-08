# Scoring functions reused from the official TAT-QA evaluator.
# Source: https://github.com/NExTplusplus/TAT-QA/blob/master/tatqa_metric.py
# MIT License
#
# Copyright (c) 2021 Fengbin Zhu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

from typing import List, Set, Tuple, Union
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.tatqa_utils import normalize_answer, is_number, to_number, scale_to_num

def _answer_to_bags(answer: Union[str, List[str], Tuple[str, ...]]) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            # if _match_numbers_if_present(gold_item, pred_item): no need to match number in tatqa
            scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (2 * precision * recall) / (precision + recall) if not (precision == 0.0 and recall == 0.0) else 0.0
    return f1


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def get_metrics(predicted: Union[str, List[str], Tuple[str, ...]],
                gold: Union[str, List[str], Tuple[str, ...]]) -> Tuple[float, float]:
    """
    Takes a predicted answer and a gold answer (that are both either a string or a list of
    strings), and returns exact match and the DROP F1 metric for the prediction.  If you are
    writing a script for evaluating objects in memory (say, the output of predictions during
    validation, or while training), this is the function you want to call, after using
    :func:`answer_json_to_strings` when reading the gold answer from the released data file.
    """
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0

    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return exact_match, f1


def extract_gold_answers(qa_annotation):
    '''
    span
    multi-span
    arithmetic (+ - * /)
    count
    date
    other
    gold answers is a list of list, each item in gold answers is a valid answer
    '''
    answer_type, scale = qa_annotation["answer_type"], qa_annotation['scale']
    answer_content = qa_annotation['answer']
    gold_answers = []
    if answer_type in ['multi-span', 'span']: # list
        assert isinstance(answer_content, list), answer_content
        gold_answers = answer_content # multi-span
    elif answer_type in ["arithmetic"]:
        gold_answers.append(str(answer_content))
    elif answer_type in ['count']:
        gold_answers.append(str(int(answer_content)))
    else:
        gold_answers.append(str(answer_content))
    return answer_type, gold_answers, scale


def metric_max_over_ground_truths(metric_fn, predictions, ground_truths):
    scores_for_ground_truths = []
    for pred in predictions:
        for ground_truth in ground_truths:
            score = metric_fn(pred, ground_truth)
            scores_for_ground_truths.append(score)
    if len(scores_for_ground_truths) == 0:
        return 0, 0
    return max(scores_for_ground_truths)



def get_answer_str(answers: list, scale: str):
    """
    :param ans_type:  span, multi-span, arithmetic, count
    :param ans_list:
    :param scale: "", thousand, million, billion, percent
    :param mode:
    :return:

    """
    sorted_ans = sorted(answers)
    ans_temp = []
    for ans in sorted_ans:
        ans_str = str(ans)
        if is_number(ans_str):
            ans_num = to_number(ans_str)
            if ans_num is None:
                if scale:
                    ans_str = ans_str + " " + str(scale)
            else:
                if '%' in ans_str: #  has been handled the answer itself is a percentage
                    ans_str = '%.4f' % ans_num
                else:
                    ans_str = '%.4f' % (round(ans_num, 2) * scale_to_num(scale))
        else:
            if scale:
                ans_str = ans_str + " " + str(scale)
        ans_temp.append(ans_str)
    return [" ".join(ans_temp)]


# handle percentage
def add_percent_pred(prediction_strings, pred_scale, pred):
    """
    to solve [pred = 0.2342] <>   [ans = 23.42 and scale == 'percent']

    :param prediction_strings:
    :param gold_ans_type:
    :param gold_scale:
    :param pred:
    :return:
    """
    if len(pred) > 1:
        return prediction_strings
    pred_str = str(pred[0])
    if pred_str is None:
        return prediction_strings
    if not pred_scale and '%' not in pred_str and is_number(pred_str): # mode only or no pred_scale num only
        pred_str = to_number(pred_str)
        if pred_str is None:
            return prediction_strings
        prediction_strings.append('%.4f' % pred_str)
    return prediction_strings
