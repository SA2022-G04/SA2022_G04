#ML_Code2seq

import numpy as np
import pandas as pd
from imblearn import over_sampling
from imblearn import pipeline
from sklearn import ensemble
from sklearn import svm
from sklearn import metrics

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

import pickle
with open("../Data/Long_Method_code2seq_train.pkl",'rb') as f:
    train = pickle.load(f)

print(train)
import pickle
with open("../Data/Long_Method_code2seq_test.pkl",'rb') as f:
    test = pickle.load(f)
print(test)
X_train, y_train = load_data("../Data/Long_Method_code2seq_train.pkl")
X_test, y_test = load_data("../Data/Long_Method_code2seq_test.pkl")
model = pipeline.make_pipeline(
    over_sampling.SMOTE(sampling_strategy=0.62, random_state=42),
    ensemble.BaggingClassifier(
        base_estimator=svm.SVC(C=0.85, random_state=42),
        random_state=42
    )
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# from sklearn.metrics import f1_score
# from sklearn.metrics import recall
# from sklearn.metrics import precision
print(metrics.classification_report(y_test, y_pred))
d = metrics.classification_report(y_test, y_pred,output_dict = True)
p = d['1']['precision']
p_c2s = round(p,2)
r = d['1']['recall']
r_c2s = round(r,2)
f = d['1']['f1-score']
f_c2s = round(f,2)


#ML|_code2vec

import numpy as np
import pandas as pd
from imblearn import combine, over_sampling, under_sampling
from imblearn import pipeline
from sklearn import ensemble
from sklearn import metrics
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
import pickle
with open("../Data/Long_Method_code2vec_train.pkl",'rb') as f:
    train = pickle.load(f)

print(train)
with open("../Data/Long_Method_code2vec_test.pkl",'rb') as f:
    test = pickle.load(f)
print(test)
X_train, y_train = load_data("../Data/Long_Method_code2vec_train.pkl")
X_test, y_test = load_data("../Data/Long_Method_code2vec_test.pkl")
model = pipeline.make_pipeline(
    over_sampling.SMOTE(sampling_strategy=0.81, random_state=42),
    ensemble.RandomForestClassifier(
        n_estimators=20, criterion="entropy", min_samples_split=5, min_samples_leaf=2, n_jobs=-1, bootstrap=1, random_state=42
    )
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
d = metrics.classification_report(y_test, y_pred,output_dict = True)
p = d['1']['precision']
p_c2v = round(p,2)
r = d['1']['recall']
r_c2v = round(r,2)
f = d['1']['f1-score']
f_c2v = round(f,2)

## ML_CuBERT
# Long Method detection

# Loading the dataset:
#
#     df - pandas DataFrame containing the whole dataset
#     df_train - pandas DataFrame containing the training set (approximately 80% of the dataset, selected by stratified sampling)
#     d_test - pandas DataFrame containing the test set (approximately 20% of the dataset, selected by stratified sampling)
#

import pickle
with open("../Data/Long_Method_CuBERT_embeddings.pkl",'rb') as f:
    cubert = pickle.load(f)
print(cubert)
import pandas as pd
import pickle
import numpy as np

pickleFile = open("../Data/Long_Method_CuBERT_embeddings.pkl", 'rb')
df = pickle.load(pickleFile)

replace_vals = {"label_x":     {"long_method": 1, "not_lm": 0}}
df = df.replace(replace_vals)

df_train = df[df['parts']=='train']
df_test = df[df['parts']=='test']
X_train = np.array([row for row in df_train['embedding']])
y_train = np.array(df_train['label_x']).astype('int')
X_test = np.array([row for row in df_test['embedding']])
y_test = np.array(df_test['label_x']).astype('int')

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from imblearn import under_sampling, combine

from sklearn import metrics

best_model = make_pipeline(combine.SMOTEENN(random_state=42,
                                                   smote=SMOTE(sampling_strategy=0.2, random_state=42),
                                                   enn=under_sampling.EditedNearestNeighbours(sampling_strategy="majority",kind_sel="mode")),
                                  XGBClassifier(learning_rate=0.6264383372848873, max_depth=76,
                                                colsample_bytree=0.016523557110861953,
                                                colsample_bylevel=0.10795972201553816, n_jobs=-1))

best_model = best_model.fit(X_train,y_train)

y_pred = best_model.predict(X_test)

print("Test report: \n", metrics.classification_report(y_test, y_pred))
d = metrics.classification_report(y_test, y_pred,output_dict = True)
p = d['1']['precision']
p_cu = round(p,2)
r = d['1']['recall']
r_cu = round(r,2)
f = d['1']['f1-score']
f_cu = round(f,2)

import xgboost as xgb

xgb.__version__


# ML_metrics

import numpy as np
import pandas as pd
from imblearn import combine, over_sampling, under_sampling
from imblearn import pipeline
from sklearn import ensemble
from sklearn import metrics
X_train_df = pd.read_excel("../Data/MLCQ_metrics_processed/train_X.xlsx")
X_train = X_train_df.values
y_train_df = pd.read_excel("../Data/MLCQ_metrics_processed/train_y.xlsx")
y_train = y_train_df["label"].values
X_test_df = pd.read_excel("../Data//MLCQ_metrics_processed/test_X.xlsx")
X_test = X_test_df.values
y_test_df = pd.read_excel("../Data/MLCQ_metrics_processed/test_y.xlsx")
y_test = y_test_df["label"].values

model = pipeline.make_pipeline(
    combine.SMOTEENN(
        random_state=42,
        smote=over_sampling.SMOTE(sampling_strategy=0.57, random_state=42),
        enn=under_sampling.EditedNearestNeighbours(kind_sel="mode", sampling_strategy="majority"),
    ),
    ensemble.RandomForestClassifier(
        n_estimators=100, criterion="entropy", n_jobs=-1, bootstrap=0, random_state=42
    )
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(metrics.classification_report(y_test, y_pred))
d = metrics.classification_report(y_test, y_pred,output_dict = True)
p = d['True']['precision']
p_metrics = round(p,2)
r = d['True']['recall']
r_metrics = round(r,2)
f = d['True']['f1-score']
f_metrics = round(f,2)


# H_metric

df = pd.read_csv("../Outputs/Tables/Long_Method_Table_6.csv")
print(df)
p_h = df.loc[4].at["Test set-P"]
r_h = df.loc[4].at["Test set-R"]
f_h = df.loc[4].at["Test set-F"]
print(df)

import matplotlib.pyplot as plt

table10 = {'Denotement': ['ML_code2vec', 'H_metrics', 'ML_code2seq', 'ML_metrics&votes', 'ML_metrics', 'ML_CuBERT'],
           'Used features': ['code2vec features', 'Code metrics', 'code2seq features',
                             'Code metrics + votes of heuristic detectors (LM1 – LM3)', 'Code metrics',
                             'CuBERT features'],
           'Approach': ['Random Forest + SMOTE', 'Heuristic-based approach (ANY)', 'Bagging (SVM classifier) + SMOTE',
                        'Random Forest + SMOTEENN', 'Random Forest + SMOTEEENN', 'XGBoost + SMOTEENN'],
           'Precision': [p_c2v, p_h, p_c2s, '0.63', p_metrics, p_cu],
           'Recall': [r_c2v, r_h, r_c2s, '0.63', r_metrics, r_cu],
           'F-measure': [f_c2v, f_h, f_c2s, 0.63, f_metrics, f_cu]
           }

table10 = pd.DataFrame(table10)
table10.to_csv('../Outputs/Tables/long_method_table10.csv')
print(table10)

import matplotlib.pyplot as plt
plot = table10.plot(x='Denotement',y='F-measure',kind='bar')
fig = plot.get_figure()
fig.savefig("../Outputs/Figures/long_method_fig6.png",bbox_inches='tight')










