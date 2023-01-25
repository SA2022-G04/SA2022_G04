import numpy as np
import pandas as pd
from imblearn import combine, over_sampling, under_sampling
from imblearn import pipeline
from sklearn import ensemble
from sklearn import metrics
import pickle
def binary_classes(smell):
    return 0 if smell == "none" else 1

def fix_data_type(vector):
    return np.asarray(vector)

def load_data(dataset):
    df = pd.read_pickle(dataset)

    df["vectors"] = df["code_vector"].apply(lambda x: fix_data_type(x))
    df["classes"] = df["smell"].apply(lambda x: binary_classes(x))

    X = list(df["vectors"].values)
    y = list(df["classes"].values)

    return X, y

# ML_code2vec
with open('../Data/God_Class_code2vec_train.pkl', 'rb') as f:
     test = pickle.load(f)
test

with open('../Data/God_Class_code2vec_test.pkl', 'rb') as f:
     test = pickle.load(f)
test
X_train, y_train = load_data("../Data/God_Class_code2vec_train.pkl")
X_test, y_test = load_data("../Data/God_Class_code2vec_test.pkl")

model = pipeline.make_pipeline(
    combine.SMOTEENN(
        random_state=42,
        smote=over_sampling.SMOTE(sampling_strategy=0.81, random_state=42),
        enn=under_sampling.EditedNearestNeighbours(kind_sel="mode", sampling_strategy="majority"),
    ),
    ensemble.RandomForestClassifier(n_estimators=460, criterion="entropy", min_samples_split=8, min_samples_leaf=2, n_jobs=-1, bootstrap=1, random_state=42)
)
model.fit(X_train, y_train )
y_pred = model.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
df = metrics.classification_report(y_test, y_pred, output_dict=True)
p = df['1']['precision']
r = df['1']['recall']
f1 = df['1']['f1-score']
p_c2v = round(p, 2)
r_c2v = round(r, 2)
f1_c2v = round(f1, 2)

# ML_code2seq
from imblearn import over_sampling
from imblearn import pipeline
from sklearn import ensemble
from sklearn import svm
from sklearn import metrics
import pickle

with open('../Data/God_Class_code2seq_train.pkl', 'rb') as f:
     test = pickle.load(f)
test

with open('../Data/God_Class_code2seq_test.pkl', 'rb') as f:
     test = pickle.load(f)
test
X_train, y_train = load_data("../Data/God_Class_code2seq_train.pkl")
X_test, y_test = load_data("../Data/God_Class_code2seq_test.pkl")
model = pipeline.make_pipeline(
    over_sampling.SMOTE(sampling_strategy=0.76, random_state=42),
    ensemble.BaggingClassifier(estimator=svm.SVC(C=0.21, random_state=42), random_state=42))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
df = metrics.classification_report(y_test, y_pred, output_dict=True)
p = df['1']['precision']
r = df['1']['recall']
f1 = df['1']['f1-score']
p_c2s = round(p, 2)
r_c2s = round(r, 2)
f1_c2s = round(f1, 2)

# ML_cuBert
# Loading the dataset:
#
# df - pandas DataFrame containing the whole dataset
# df_train - pandas DataFrame containing the training set (approximately 80% of the dataset, selected by stratified sampling)
# d_test - pandas DataFrame containing the test set (approximately 20% of the dataset, selected by stratified sampling)
import pickle
import pandas as pd
import pickle
import numpy as np
from sklearn import ensemble
from sklearn import svm
from imblearn import combine, under_sampling, over_sampling
from sklearn import metrics

with open('../Data/God_Class_CuBERT_embeddings.pkl', 'rb') as f:
     test = pickle.load(f)
test

pickleFile = open("../Data/God_Class_CuBERT_embeddings.pkl", 'rb')
df = pickle.load(pickleFile)
replace_vals = {"label":     {"crit_blob": 1, "not_blob": 0}}
df = df.replace(replace_vals)

df_train = df[df['parts'] == 'train']
y_train = df_train['label']
df_test = df[df['parts'] == 'test']
y_test = df_test['label']
# X_train: using CuBERT embedding as features. Each row is represented as a 1024-dim vector. We created this vector by applying a pre-trained CuBERT Java model https://github.com/google-research/google-research/tree/master/cubert to code snippets.
X_train = np.array([row for row in df_train['embedding']])
y_train = np.array(df_train['label'])
X_test = np.array([row for row in df_test['embedding']])
y_test = np.array(df_test['label'])
# Training the model on the train portion of the Data

model = ensemble.BaggingClassifier(
    estimator=svm.SVC(
        C=0.010884125795534865,
        kernel="rbf",
        gamma="scale",
        random_state=42
    ),
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

resampler = combine.SMOTEENN(
        random_state=42,
        smote=over_sampling.SMOTE(
            sampling_strategy="minority", random_state=42
        ),
        enn=under_sampling.EditedNearestNeighbours(
            sampling_strategy="majority",
            kind_sel="mode"
        )
    )

X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)

model.fit(X_resampled, y_resampled)
# Applying the model on the test portion of the Data
y_pred = model.predict(X_test)
print("Test report: \n", metrics.classification_report(y_test, y_pred))
df = metrics.classification_report(y_test, y_pred, output_dict=True)
p = df['1']['precision']
r = df['1']['recall']
f1 = df['1']['f1-score']
p_cubert = round(p, 2)
r_cubert = round(r, 2)
f1_cubert = round(f1, 2)

# Ml_metrics
import numpy as np
import pandas as pd
from imblearn import combine, over_sampling, under_sampling
from imblearn import pipeline
from sklearn import ensemble
from sklearn import metrics
X_train_df = pd.read_excel('../Data/MLCQ_metrics_processed_/MLCQ_train_X.xlsx')
X_train_df
X_train = X_train_df.values
y_train_df = pd.read_excel("../Data/MLCQ_metrics_processed_/MLCQ_train_y.xlsx")
y_train_df
y_train = y_train_df["label"].values
X_test_df = pd.read_excel("../Data/MLCQ_metrics_processed_/MLCQ_test_X.xlsx")
X_test_df
X_test = X_test_df.values
y_test_df = pd.read_excel("../Data/MLCQ_metrics_processed_/MLCQ_test_y.xlsx")
y_test_df
y_test = y_test_df["label"].values
model = pipeline.make_pipeline(
    combine.SMOTEENN(
         random_state=42,
         smote=over_sampling.SMOTE(sampling_strategy=0.7, random_state=42),
         enn=under_sampling.EditedNearestNeighbours(kind_sel="mode", sampling_strategy="majority"),
     ), ensemble.RandomForestClassifier(n_estimators=100, criterion="entropy", min_samples_leaf=2, min_samples_split=5, n_jobs=-1, bootstrap=1, random_state=42)
 )
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
df = metrics.classification_report(y_test, y_pred, output_dict=True)
p = df['True']['precision']
r = df['True']['recall']
f1 = df['True']['f1-score']
p_ml_metric = round(p, 2)
r_ml_metric = round(r, 2)
f1_ml_metric = round(f1, 2)

# Ml_metrics_and_heuristics
import pandas as pd
import pickle
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn import combine, under_sampling, over_sampling
from sklearn import metrics

with open('../Data/God_Class_metrics_and_heuristics.pkl', 'rb') as f:
     test = pickle.load(f)
test

pickleFile = open("../Data/God_Class_metrics_and_heuristics.pkl", 'rb')
df = pickle.load(pickleFile)
df_train = df[df['parts'] == 'train']
df_test = df[df['parts'] == 'test']
X_train = np.array(df_train.drop(['parts', 'label', 'file', 'type', 'ALL', 'ANY', 'MAJOR VOTES', '([ATFD] > 2) & ([WMC] ≥ 47) & ([TCC] < 0.33)', '([WMC] ≥ 47) & ([TCC] < 0.3) & ([ATFD] > 5)'], axis=1))
y_train = np.array(df_train['label'])
X_test = np.array(df_test.drop(['parts', 'label', 'file', 'type', 'ALL', 'ANY', 'MAJOR VOTES', '([ATFD] > 2) & ([WMC] ≥ 47) & ([TCC] < 0.33)', '([WMC] ≥ 47) & ([TCC] < 0.3) & ([ATFD] > 5)'], axis=1))
y_test = np.array(df_test['label'])

std_scaler = StandardScaler()
sc = std_scaler.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

model = RandomForestClassifier(
                n_estimators=360,
                max_features="log2",
                min_samples_split=8,
                min_samples_leaf=4,
                bootstrap=False,
                criterion="entropy",
                random_state=42,
                n_jobs=-1
            )

resampler = combine.SMOTEENN(
        random_state=42,
        smote=over_sampling.SMOTE(
            sampling_strategy=0.8,
            random_state=42),
        enn=under_sampling.EditedNearestNeighbours(
            sampling_strategy="majority",
            kind_sel="mode"
        )
    )

X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)

model.fit(X_resampled, y_resampled)

y_pred = model.predict(X_test)

print("Test report: \n", metrics.classification_report(y_test, y_pred))
df = metrics.classification_report(y_test, y_pred, output_dict=True)
p = df['1']['precision']
r = df['1']['recall']
f1 = df['1']['f1-score']
p_ml_M_H = round(p,2)
r_ml_M_H = round(r,2)
f1_ml_M_H = round(f1,2)

# H_metrics
df = pd.read_csv('../Outputs/Tables/God_Class_Table_5.csv')
df
p_H_metrics = df.loc[7].at["Test set-P"]
r_H_metrics = df.loc[7].at["Test set-R"]
f1_H_metrics = df.loc[7].at["Test set-F"]

# drawing Table
Table9 = pd.DataFrame(columns=['Denotement', 'Features', 'Approach', 'Precision', 'Recall', 'F-measure'])
data = {'Denotement':['ML_code2vec', 'ML_code2seq', 'H_metrics', 'ML_metrics', 'ML_metrics&votes', 'ML_CuBERT'],
        'Features': ['code2vec features', 'code2seq features', 'Code metrics', 'Code metrics', 'Code metrics + votes of heuristic detectors (GC3 – GC8)', 'CuBERT features'],
        'Approach': ['Random Forest + SMOTEENN', 'Random Forest + SMOTEENN', 'Heuristic detector (GC8)', 'Random Forest + SMOTEENN', 'Random Forest + SMOTEENN', 'Bagging (SVM classifier) + SMOTEENN'],
        'Precision': [p_c2v, p_c2s, p_H_metrics, p_ml_metric, p_ml_M_H, p_cubert],
        'Recall': [r_c2v, r_c2s, r_H_metrics, r_ml_metric, r_ml_M_H, r_cubert],
        'F-measure': [f1_c2v, f1_c2s, f1_H_metrics, f1_ml_metric, f1_ml_M_H, f1_cubert]}
Table9 = pd.DataFrame(data)
Table9.to_csv('../Outputs/Tables/God_Class_Table9.csv')
Table9

# drawing plot
import matplotlib.pyplot as plt
plot = Table9.plot(x='Denotement', y='F-measure', kind='bar')
plt.savefig("../Outputs/Figures/God_Class_Fig_6.jpg", dpi=100, bbox_inches='tight')

