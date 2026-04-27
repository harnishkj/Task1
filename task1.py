import pandas as pd
import numpy as np
import re
import ast
from scipy.stats import wilcoxon


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import hstack, csr_matrix

import nltk

#Text preprocessing

def remove_html(text):
    return re.compile(r'<.*?>').sub(r'', text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

#Structural feature extraction
def extract_structural_features(text):
    text_str = str(text)
    features = []

    # Stack trace pattern (e.g. Java/Python tracebacks)
    has_stack_trace = 1 if re.search(r'at [\w\.\$]+\([\w\.]+:\d+\)', text_str) else 0
    features.append(has_stack_trace)

    # Code blocks (markdown or HTML)
    has_code_block = 1 if ('```' in text_str or '<code>' in text_str) else 0
    features.append(has_code_block)

    # Error keyword count
    error_keywords = len(re.findall(
        r'\b(error|exception|crash|fail|traceback|bug|issue|wrong|incorrect|broken)\b',
        text_str.lower()
    ))
    features.append(error_keywords)

    # Version number presence
    has_version = 1 if re.search(r'v?\d+\.\d+(\.\d+)?', text_str) else 0
    features.append(has_version)

    # Word count
    word_count = len(text_str.split())
    features.append(word_count)

    # URL presence
    has_url = 1 if re.search(r'https?://', text_str) else 0
    features.append(has_url)

    # "Expected/actual/reproduce" pattern (classic bug report structure)
    has_expected_actual = 1 if re.search(
        r'(expected|actual|reproduce|steps to reproduce)', text_str.lower()
    ) else 0
    features.append(has_expected_actual)

    return features

#Load data

project = 'incubator-mxnet'  # Change to dataset name
path = f"datasets/{project}.csv"

pd_all = pd.read_csv(path)
pd_all = pd_all.sample(frac=1, random_state=999)

pd_all['Title+Body'] = pd_all.apply(
    lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
    axis=1
)

pd_tplusb = pd_all.rename(columns={
    "Unnamed: 0": "id",
    "class": "sentiment",
    "Title+Body": "text"
})
pd_tplusb.to_csv('Title+Body.csv', index=False, columns=["id", "Number", "sentiment", "text"])

#Configure parameters

datafile = 'Title+Body.csv'
REPEAT = 10
out_csv_name = f"results/{project}_SVM.csv"

#Read and clean data

data = pd.read_csv(datafile).fillna('')
text_col = 'text'

raw_text = data[text_col].copy()

# Clean text for TF-IDF
data[text_col] = data[text_col].apply(remove_html)
data[text_col] = data[text_col].apply(remove_emoji)
data[text_col] = data[text_col].apply(clean_str)


accuracies = []
precisions = []
recalls    = []
f1_scores  = []
auc_values = []

#Repeated train/test splits
for repeated_time in range(REPEAT):
    indices = np.arange(data.shape[0])
    train_index, test_index = train_test_split(
        indices, test_size=0.2, random_state=repeated_time
    )

    train_text = data[text_col].iloc[train_index]
    test_text  = data[text_col].iloc[test_index]

    train_raw  = raw_text.iloc[train_index]
    test_raw   = raw_text.iloc[test_index]

    y_train = data['sentiment'].iloc[train_index]
    y_test  = data['sentiment'].iloc[test_index]

    #TF-IDF
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    X_train_tfidf = tfidf.fit_transform(train_text)
    X_test_tfidf  = tfidf.transform(test_text)

    #Structural features
    train_struct = csr_matrix(np.array([extract_structural_features(t) for t in train_raw]))
    test_struct  = csr_matrix(np.array([extract_structural_features(t) for t in test_raw]))

    #Combine TF-IDF + structural features
    X_train = hstack([X_train_tfidf, train_struct])
    X_test  = hstack([X_test_tfidf,  test_struct])

    #Scale features
    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    #SVM
    best_clf = CalibratedClassifierCV(LinearSVC(C=1, class_weight='balanced', max_iter=5000), cv=3)
    best_clf.fit(X_train, y_train)
    
    #Predictions + metrics
    y_pred = best_clf.predict(X_test)
    y_prob = best_clf.predict_proba(X_test)[:, 1]  # probabilities for AUC

    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
    recalls.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
    f1_scores.append(f1_score(y_test, y_pred, average='macro', zero_division=0))

    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
    auc_values.append(auc(fpr, tpr))

#Save results

final_accuracy  = np.mean(accuracies)
final_precision = np.mean(precisions)
final_recall    = np.mean(recalls)
final_f1        = np.mean(f1_scores)
final_auc       = np.mean(auc_values)

print("=== SVM + TF-IDF + Structural Features Results ===")
print(f"Number of repeats:     {REPEAT}")
print(f"Average Accuracy:      {final_accuracy:.4f}")
print(f"Average Precision:     {final_precision:.4f}")
print(f"Average Recall:        {final_recall:.4f}")
print(f"Average F1 score:      {final_f1:.4f}")
print(f"Average AUC:           {final_auc:.4f}")

try:
    pd.read_csv(out_csv_name, nrows=1)
    header_needed = False
except:
    header_needed = True

df_log = pd.DataFrame({
    'repeated_times': [REPEAT],
    'Accuracy':       [final_accuracy],
    'Precision':      [final_precision],
    'Recall':         [final_recall],
    'F1':             [final_f1],
    'AUC':            [final_auc],
    'CV_list(AUC)':   [str(auc_values)]
})

df_log.to_csv(out_csv_name, mode='a', header=header_needed, index=False)
print(f"\nResults saved to: {out_csv_name}")

#p values

projects = ['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe']

for project in projects:
    nb_df  = pd.read_csv(f"results/{project}_NB.csv")
    svm_df = pd.read_csv(f"results/{project}_SVM.csv")

    nb_aucs  = ast.literal_eval(nb_df.iloc[-1]['CV_list(AUC)'])
    svm_aucs = ast.literal_eval(svm_df.iloc[-1]['CV_list(AUC)'])

    stat, p = wilcoxon(nb_aucs, svm_aucs)
    print(f"{project}: p-value = {p:.4f}")
