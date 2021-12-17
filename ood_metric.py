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

import datasets
import torch
import torch.nn.functional as F


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
class NewMetric(datasets.Metric):
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
                'predictions': datasets.Value('string'),
                'references': datasets.Value('string'),
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
        logits, pooled, is_ind = predictions
        pdb.set_trace()
        accuracy = sum(i == j for i, j in zip(logits, references)) / len(predictions)
        
        if is_sigmoid == False:
            softmax_score = F.softmax(logits, dim=-1).max(-1)[0]
        else:
            softmax_score, sig_pred = F.sigmoid(logits).max(-1)

        maha_score = []

        for c in self.label_id_list:
            centered_pooled = pooled - self.class_mean[c].unsqueeze(0)
            ms = torch.diag(centered_pooled @ self.class_var @ centered_pooled.t())
            maha_score.append(ms)
        maha_score = torch.stack(maha_score, dim=-1)

        maha_score, pred = maha_score.min(-1)
        maha_score = -maha_score
        if ind == True:
            if is_sigmoid == True:
                correct = (references == sig_pred).float().sum()
            else:
                correct = (references == pred).float().sum()
        else:
            correct = 0

        
        norm_pooled = F.normalize(pooled, dim=-1)
        cosine_score = norm_pooled @ self.norm_bank.t()
        cosine_score = cosine_score.max(-1)[0]

        energy_score = torch.logsumexp(logits, dim=-1)



        return {
            "accuracy": accuracy,
            "maha_acc": maha_acc,
            "AUROC(softmax)": auroc_softmax,
            "FPR-95(softmax)": fpr95_softmax,
            "AUROC(maha)": auroc_maha,
            "FPR-95(maha)": fpr95_maha,
            "AUROC(cosine)": auroc_cosine,
            "FPR-95(cosine)": fpr95_cosine,
            "AUROC(energy)": auroc_energy,
            "FPR-95(energy)": fpr95_energy,
        }