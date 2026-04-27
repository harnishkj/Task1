Replication
Setup
1. Ensure the `datasets/` folder contains:
   - pytorch.csv
   - tensorflow.csv
   - keras.csv
   - incubator-mxnet.csv
   - caffe.csv

2. Create a `results/` folder if it does not exist:
       mkdir results

3. Install dependencies:
       pip install pandas numpy scikit-learn scipy nltk

Replicating Results

1. Run the Naive Bayes baseline
In `br_classification.py`, set `project` to each of the five datasets
and run the script each time:

    python br_classification.py

Repeat for: pytorch, tensorflow, keras, incubator-mxnet, caffe

2. Run the SVM model
In `task1.py`, set `project` to each of the five datasets and run:

    python task1.py

Repeat for: pytorch, tensorflow, keras, incubator-mxnet, caffe

3. Compute p-values
Run the following script to reproduce the Wilcoxon significance test results:

    import pandas as pd
    import ast
    from scipy.stats import wilcoxon

    projects = ['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe']

    for project in projects:
        nb_df  = pd.read_csv(f"results/{project}_NB.csv")
        svm_df = pd.read_csv(f"results/{project}_SVM.csv")
        nb_aucs  = ast.literal_eval(nb_df.iloc[-1]['CV_list(AUC)'])
        svm_aucs = ast.literal_eval(svm_df.iloc[-1]['CV_list(AUC)'])
        stat, p = wilcoxon(nb_aucs, svm_aucs)
        print(f"{project}: p-value = {p:.4f}")

