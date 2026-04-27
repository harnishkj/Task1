
Overview
This tool classifies GitHub issue reports as bug reports (1) or non-bug reports (0)
using two models: a Naive Bayes baseline and an SVM classifier.

Files
- `br_classification.py` — Naive Bayes baseline
- `task1.py` — SVM model with TF-IDF and structural features

Selecting a Dataset
In both scripts, change the `project` variable to one of:
`pytorch`, `tensorflow`, `keras`, `incubator-mxnet`, `caffe`

In `task1.py` (line 90):
    project = 'pytorch'

In `br_classification.py` (line 72):
    project = 'pytorch'

Running
    python br_classification.py
    python task1.py

 Output
Results are saved to:
- `results/{project}_NB.csv` — Naive Bayes results
- `results/{project}_SVM.csv` — SVM results

Each CSV contains: Accuracy, Precision, Recall, F1, AUC, and the
individual AUC values across 10 repeated splits.