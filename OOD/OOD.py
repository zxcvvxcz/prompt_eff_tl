# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Add a description here."""

import pdb
from typing import Optional, Tuple, Union
import datasets
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{huggingface:metric,
title = {A great new metric},
authors={huggingface, Inc.},
year={2020}
}
"""

# TODO: Add description of the metric here
_DESCRIPTION = """\
Evaluate AUROC, fpr95 for a bunch of metrics (energy, mahalanobis, softmax, cosine).
"""


# TODO: Add description of the arguments of the metric here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    accuracy: description of the first score,
    another_score: description of the second score,
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.
    >>> my_new_metric = datasets.load_metric("my_new_metric")
    >>> results = my_new_metric.compute(references=[0, 1], predictions=[0, 1])
    >>> print(results)
    {'accuracy': 1.0}
"""

# TODO: Define external resources urls if needed
BAD_WORDS_URL = "http://url/to/external/resource/bad_words.txt"


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class OOD(datasets.Metric):
    """TODO: Short description of my metric."""
        
    def _info(self):
        # TODO: Specifies the datasets.MetricInfo object
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'predictions': datasets.Value('float32'),
                'references': datasets.Value('int16'),
            }),
            # Homepage of the metric for documentation
            homepage="http://metric.homepage",
            # Additional links to the codebase or references
            codebase_urls=["http://github.com/path/to/codebase/of/new_metric"],
            reference_urls=["http://path.to.reference.url/new_metric"]
        )

    def _download_and_prepare(self, dl_manager):
        """Optional: download external resources useful to compute the scores"""
        # TODO: Download external resources if needed
        # bad_words_path = dl_manager.download_and_extract(BAD_WORDS_URL)
        # self.bad_words = set([w.strip() for w in open(bad_words_path, "r", encoding="utf-8")])
        ...

    def _compute(self, predictions, references, is_sigmoid=False):
        """Returns the scores"""
        # TODO: Compute the different scores of the metric
        
        # accuracy = sum(i == j for i, j in zip(logits, references)) / len(predictions)
        

        auroc, fpr_95 = get_auroc(references, predictions), get_fpr_95(references, predictions)

        return {
           f"AUROC({self.config_name})": auroc,
            f"FPR-95({self.config_name})": fpr_95,
        }
        
        
def get_auroc(key, prediction):
    new_key = np.copy(key)
    new_key[new_key == 0] = 0
    new_key[new_key > 0] = 1
    return roc_auc_score(new_key, prediction)


def get_fpr_95(key, prediction, return_indices=False):
    new_key = np.copy(key)
    new_key[new_key == 0] = 0
    new_key[new_key > 0] = 1
    prediction = np.array(prediction, dtype=np.float32)
    if return_indices:
        score, indices = fpr_and_fdr_at_recall(new_key, prediction, return_indices)
        return score, indices
    else:
        score = fpr_and_fdr_at_recall(new_key, prediction, return_indices)
        return score

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, return_indices, recall_level=0.95, pos_label=1.):
    y_true = (y_true == pos_label)

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    y_wrong_indices = desc_score_indices[np.where(y_true == False)[0]]
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps


    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1) # last ind부터 역순으로
    # recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]
    recall, fps = np.r_[recall[sl], 1], np.r_[fps[sl], 0]

    cutoff = np.argmin(np.abs(recall - recall_level))

    if return_indices:
        return fps[cutoff] / (np.sum(np.logical_not(y_true))), y_wrong_indices
    else:
        return fps[cutoff] / (np.sum(np.logical_not(y_true)))
