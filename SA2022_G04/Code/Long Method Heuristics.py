import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os

#LM_1 is from Fard et al. (2013): Long Method has many lines of code.
def LM_1(df):
    df['LM_1'] = df['loc'] > 50
    df['LM_1'].replace({False:0, True:1},inplace=True)

#LM_2 is from Souza et al. (2017): Long Method is huge, complex and has a high number of nested blocks.
def LM_2(df):
    df['LM_2'] = (df['loc'] > 30) & (df['wmc'] > 4) & (df['maxNestedBlocksQty'] > 3)
    df['LM_2'].replace({False:0, True:1},inplace=True)

#LM_3 is from Liu et al. (2011): Long Method has many lines of code or is highly complex.
def LM_3(df):
    df['LM_3'] = (df['loc'] > 50) | (df['wmc'] > 10)
    df['LM_3'].replace({False:0, True:1},inplace=True)

#ALL – LM_1, LM_2, and LM_3
def all_heuristics(df):
    df['ALL'] = df['LM_1'] & df['LM_2'] & df['LM_3']

#ANY – LM_1, LM_2, or LM_3
def any_heuristics(df):
    df['ANY'] = df['LM_1'] | df['LM_2'] | df['LM_3']

#Precision, recall, and F1-measure of the minority (smell) class (denoted as 1).
def precision(true, predicted):
    tn, fp, fn, tp = confusion_matrix(true, predicted).ravel()
    return tp/(tp+fp)

def recall(true, predicted):
    tn, fp, fn, tp = confusion_matrix(true, predicted).ravel()
    return tp/(tp+fn)

def fmeasure(true, predicted):
    p = precision(true, predicted)
    r = recall(true, predicted)
    return 2*p*r/(p+r)

#Weighted vote – the model calculates the probability of the code sample suffering from the smell as a weighted vote of the individual classifiers. We use the F1-measure achieved on the training set as the weight and apply the softmax function to normalize weights (so they sum to 1).
# applies softmax for each sets of scores in input:x.
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_weights(df):
    true = df['label']
    heuristics = ['LM_1', 'LM_2', 'LM_3']
    weights = np.zeros(len(heuristics))
    for i in range(len(heuristics)):
        predicted = df[heuristics[i]]
        weights[i] = fmeasure(true, predicted)
    print("weights:", weights)
    return weights


def weighted_vote(df):
    pd.options.mode.chained_assignment = None  # default='warn'

    heuristic_fmeasures = get_weights(df)
    weights = softmax(heuristic_fmeasures)
    df['Weighted_Vote_probability'] = df['LM_1'] * weights[0] + df['LM_2'] * weights[1] + df['LM_3'] * weights[2]
    is_smell = df['Weighted_Vote_probability'] > 0.5
    print("is smell: ", is_smell)
    df['Weighted_Vote'] = 0
    df['Weighted_Vote'][is_smell] = 1
    print("Weighted Vote: ", df['Weighted_Vote'])

#applies all heuristics to the df DataFrame
def apply_heuristics(df):
    LM_1(df)
    LM_2(df)
    LM_3(df)
    all_heuristics(df)
    any_heuristics(df)
    weighted_vote(df)

#Part : "train", "test" or "all" Heuristic : GC_1, GC_2, ..., GC_8, ALL, ANY, Weighted_Vote
def eval_calculation(df, heuristic):
    true = df['label']
    predicted = df[heuristic]

    p = precision(true, predicted)
    r = recall(true, predicted)
    f = fmeasure(true, predicted)

    return p, r, f

#Part : "train", "test" or "all" Heuristic : LM_1, LM_2, LM_3, ALL, ANY, Weighted_Vote
def print_result(df, heuristic, part):
    p, r, f = eval_calculation(df, heuristic)

    print("Heuristic", heuristic, "on", part, ":", "Precision: %.2f" % p, "; Recall: %.2f" % r, "; F-measure: %.2f" % f)

#Result of Table 6:
def result(df, df_train, df_test, index, data):
    rule = ['MLOC > 50', 'MLOC > 30 & VG > 4 & NBD > 3', 'MLOC > 50 | VG > 10', 'ALL', 'ANY', 'Weighted Vote']
    trainP = []
    trainR = []
    trainF = []
    testP = []
    testR = []
    testF = []
    allP = []
    allR = []
    allF = []
    for i in range(len(index)):
        p1, r1, f1 = eval_calculation(df_train, index[i])
        trainP.append(round(p1, 2))
        trainR.append(round(r1, 2))
        trainF.append(round(f1, 2))

        p2, r2, f2 = eval_calculation(df_test, index[i])
        testP.append(round(p2, 2))
        testR.append(round(r2, 2))
        testF.append(round(f2, 2))

        p3, r3, f3 = eval_calculation(df, index[i])
        allP.append(round(p3, 2))
        allR.append(round(r3, 2))
        allF.append(round(f3, 2))

        p1 = p2 = p3 = r1 = r2 = r3 = f1 = f2 = f3 = 0

    data['Rule specification'] = rule
    data['Training set-P'] = trainP
    data['Training set-R'] = trainR
    data['Training set-F'] = trainF
    data['Test set-P'] = testP
    data['Test set-R'] = testR
    data['Test set-F'] = testF
    data['All-P'] = allP
    data['All-R'] = allR
    data['All-F'] = allF

    df = pd.DataFrame(data, index)

    return df

#Visualization:
def visualization(df2):
    barWidth = 0.25
    fig = plt.figure(figsize=(12, 8))
    preci = []
    rec = []
    fmeas = []
    heuristics = []
    df3 = df2[['Test set-P', 'Test set-R', 'Test set-F']]
    df_visual = df3.sort_values('Test set-F')

    preci = df_visual['Test set-P']
    rec = df_visual['Test set-R']
    fmeas = df_visual['Test set-F']
    heuristics = df_visual.index

    br1 = np.arange(len(preci))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    plt.bar(br1, preci, color='b', width=barWidth, label='Precision')
    plt.bar(br2, rec, color='g', width=barWidth, label='Recall')
    plt.bar(br3, fmeas, color='orange', width=barWidth, label='F-measure')

    plt.xlabel('Long Method Heuristics', fontweight='bold', fontsize=15)
    plt.ylabel('Values', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(preci))], heuristics)

    plt.legend()
    plt.show()

    fig.savefig('../Outputs/Figures/Heuristic_based_approaches_Long_Method_test_set_Fig_5.png', dpi=fig.dpi)

    plt.close()


def main():
    """
    df - pandas DataFrame containing the whole dataset
    df_train - pandas DataFrame containing the training set (approximately 80% of the dataset, selected by stratified sampling)
    d_test - pandas DataFrame containing the test set (approximately 20% of the dataset, selected by stratified sampling)
    """
    df = pd.read_excel("../Data/Long_Method_code_metrics_values.xlsx")
    new_labels = {"label": {"critical": 1, "major": 1, "minor": 1, "none": 0}}
    df = df.replace(new_labels)

    df_train = df[df['parts'] == 'train']
    y_train = df_train['label']
    df_test = df[df['parts'] == 'test']
    y_test = df_test['label']

    print("The first 5 rows of the training dataframe:\n", df_train.head())
    print("*****************************************************************")
    print("The labels of the first 5 lines of the training dataframe:\n", y_train.head())
    print("*****************************************************************")
    print("The first 5 rows of the testing dataframe:\n", df_test.head())
    print("*****************************************************************")
    print("The labels of the first 5 lines of the testing dataframe:\n", y_test.head())

    print("Number of examples:", df.shape[0])
    print("Number of train examples:", df_train.shape[0], "; positive:", df_train[y_train == 1].shape[0],
          "; negative: ", df_train[y_train == 0].shape[0])
    print("Number of test examples:", df_test.shape[0], "; positive:", df_test[y_test == 1].shape[0], "; negative: ",
          df_test[y_test == 0].shape[0])

    #We show weights here to make sure that our code works.
    apply_heuristics(df)
    apply_heuristics(df_train)
    apply_heuristics(df_test)

    #Printing the performance of each heuristic:
    print_result(df_train, 'ANY', 'train')
    print_result(df_test, 'ANY', 'test')
    print_result(df, 'ANY', 'all')

    print_result(df_train, 'Weighted_Vote', 'train')
    print_result(df_test, 'Weighted_Vote', 'test')
    print_result(df, 'Weighted_Vote', 'all')

    print_result(df_train, 'ALL', 'train')
    print_result(df_test, 'ALL', 'test')
    print_result(df, 'ALL', 'all')

    print_result(df_train, 'LM_3', 'train')
    print_result(df_test, 'LM_3', 'test')
    print_result(df, 'LM_3', 'all')

    print_result(df_train, 'LM_2', 'train')
    print_result(df_test, 'LM_2', 'test')
    print_result(df, 'LM_2', 'all')

    print_result(df_train, 'LM_1', 'train')
    print_result(df_test, 'LM_1', 'test')
    print_result(df, 'LM_1', 'all')

    #Show dataframe:
    index = ['LM_1', 'LM_2', 'LM_3', 'ALL', 'ANY', 'Weighted_Vote']
    data = {'Rule specification': [], 'Training set-P': [], 'Training set-R': [], 'Training set-F': [],
            'Test set-P': [], 'Test set-R': [], 'Test set-F': [],
            'All-P': [], 'All-R': [], 'All-F': []}
    df2 = result(df, df_train, df_test, index, data)
    print("Table 6: \n", df2)

    # Change dataframe to csv format and save it:
    df2.to_csv(os.path.join('../Outputs/Tables', 'Long_Method_Table_6.csv'))

    # Fig 5:
    print("Fig 5: \n")
    visualization(df2)


main()