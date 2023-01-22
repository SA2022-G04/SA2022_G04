#library:
import pandas as pd
import os

#Result for Table 7:
def result(df_god, df_method, index, data):
    app = ['[ATFD] > 2 & [WMC] ≥ 47 & [TCC] < 0.33', '[WMC] ≥ 47 & [TCC] < 0.3 & [ATFD] > 5', '[NOM] > 15 | [NOF] > 15',
           '[CLOC] > 750 | [NOM] + [NOF] > 20',
           '[NOM] + [NOF] > 20', '[LCOM] ≥ 0.725 & [WMC] ≥ 34 & [NOF] ≥ 8 & [NOM] ≥ 14',
           '[NOM] > 20 | [NOF] > 9 | [CLOC] > 750', '[CLOC] > 100 | [WMC] > 20',
           'MLOC > 50', 'MLOC > 30 & VG > 4 & NBD > 3', 'MLOC > 50 | VG > 10']

    ex_descript = [
        'A single open-source project.The authors do not report performance measures. Instead, they select a few top code smells(detected by their approach) and discuss why they constitute code smells.',
        'Three open-source projects.The original developers annotated the smells. The authors inform the precision and recall for each project separately. However, as they have reported the exact number of TP, FP, and FN perproject, we calculated the aggregated precision and recall, which we present here.Annotations are not publicly available.',
        'A single open-source project.The authors do not report performance measures. Instead, they perform a case study by measuring the number of bug reports issued for the code smells detected by their approach andshow that the largest code smell candidate has the largest number of filed bug reports.',
        'The authors consider nine JavaScript applications chosen for manual checking as they do not have a large codebase. The authors report a total of 15 God Classes in this codebase.The dataset is manually annotated. However, the authors do not specify if multiple annotators examined each code sample. They do not state the annotator’s expertise or understanding of the code smells.The annotations are not publicly available.',
        'The dataset consists of 11 manually annotated open-source systems.The dataset is manually annotated.Five annotators examined a single system using Brown’s and Fowler’s books as references. The authors report that their approach achieves 100% recall and 89.6% precision on this single system.The authors only report the precision for the remaining ten systems as it is too costly to calculate recall. As the authors report the precision per system, we report the average precision here. The paper states that the dataset is publicly available; however, we could not access the dataset link from the paper when we performed our experiments.',
        'The authors examined 12 software systems. The authors compare their approach to the results obtained by the existing tools JDeodorant and JSPiRIT. Additionally, annotators examined two of those systems manually using Fowler’s book as a reference. They do not specify whether multiple annotators analyzed the same code snippets. Here, we report the average precision and recall for those two systems. The dataset is not publicly available.',
        'The dataset consisted of 323 Java classes. The authors report they tested the used heuristics by using known bad smell source code from the “Refactoring: Improving the Design of existing Code” book. The approach had 100% accuracy for some smells for this source code. However, they do not report the accuracy for the Large Class code smell.',
        'The authors present a motivating example in which they apply their approach to a single systems’ source code.',
        'The authors consider nine JavaScript applications chosen for manual checking as they do not have a large codebase. The authors report a total of 25 Long Methods in this codebase. The dataset is manually annotated. However, the authors do not specify if multiple annotators examined each code sample. They do not state the annotator’s expertise or understanding of the code smells. The annotations are not publicly available.',
        'The authors examined 12 software systems. The authors compare their approach to the results obtained by the existing tools JDeodorant and JSPiRIT. Additionally, annotators examined two of those systems manually using Fowler’s book as a reference. They do not specify whether multiple annotators analyzed the same code snippets. Here, we report the average precision and recall for those two systems. The dataset is not publicly available.',
        'The authors present a motivating example in which they apply their approach to a single systems’ source code.']

    original_p = ['N/A', 0.5, 'N/A', 0.78, 0.89, 0.75, 'N/A', 'N/A', 1, 0.29, 'N/A']
    original_r = ['N/A', 0.38, 'N/A', 0.94, 'N/A', 0.45, 'N/A', 'N/A', 1, 1, 'N/A']

    MLCQ_p = []
    MLCQ_r = []

    god_p = df_god['Test set-P'][:8].values
    god_r = df_god['Test set-R'][:8].values

    long_p = df_method['Test set-P'][:3].values
    long_r = df_method['Test set-R'][:3].values

    for i in range(8):
        MLCQ_p.append(god_p[i])
        MLCQ_r.append(god_r[i])

    for j in range(3):
        MLCQ_p.append(long_p[j])
        MLCQ_r.append(long_r[j])

    data['Approach'] = app
    data['Experiment description'] = ex_descript
    data['MLCQ dataset-P'] = MLCQ_p
    data['MLCQ dataset-R'] = MLCQ_r
    data['Original study-P'] = original_p
    data['Original study-R'] = original_r

    df = pd.DataFrame(data, index)

    return df

#Result for table 8:
def result_2(df_god, df_method, index_2, data_2):
    god_p_list = df_god['All-P'][:8].values
    god_r_list = df_god['All-R'][:8].values

    long_p_list = df_method['All-P'][:3].values
    long_r_list = df_method['All-R'][:3].values

    MLCQ_p_g = []
    MLCQ_r_g = []
    MLCQ_p_l = []
    MLCQ_r_l = []

    for i in range(8):
        MLCQ_p_g.append(god_p_list[i])
        MLCQ_r_g.append(god_r_list[i])

    for j in range(3):
        MLCQ_p_l.append(long_p_list[j])
        MLCQ_r_l.append(long_r_list[j])

    god_p = round((sum(MLCQ_p_g) / 8) * 100, 1)
    god_r = round((sum(MLCQ_r_g) / 8) * 100, 1)

    long_p = round((sum(MLCQ_p_l) / 3) * 100, 1)
    long_r = round((sum(MLCQ_r_l) / 3) * 100, 1)

    god_class_ave_p = [god_p, 64.8]
    god_class_ave_r = [god_r, 14]
    long_method_ave_p = [long_p, 74.3]
    long_method_ave_r = [long_r, 45.8]

    data_2['God Class-Average precision%'] = god_class_ave_p
    data_2['God Class-Average recall%'] = god_class_ave_r
    data_2['Long Method-Average precision%'] = long_method_ave_p
    data_2['Long Method-Average recall%'] = long_method_ave_r

    df_2 = pd.DataFrame(data_2, index_2)

    return df_2

def main():
    #Read result of table 5:
    df_god = pd.read_csv('../Outputs/Tables/God_Class_Table_5.csv')
    print("df_god \n", df_god)

    #Read result of table 6:
    df_method = pd.read_csv('../Outputs/Tables/Long_Method_Table_6.csv')
    print("df_method \n", df_method)

    #Show result:
    index = ['GC_1', 'GC_2', 'GC_3', 'GC_4', 'GC_5', 'GC_6', 'GC_7', 'GC_8', 'LM_1', 'LM_2', 'LM_3']
    data = {'Approach': [], 'Experiment description': [], 'MLCQ dataset-P': [], 'MLCQ dataset-R': [],
            'Original study-P': [], 'Original study-R': []}

    df = result(df_god, df_method, index, data)
    print("Table 7: \n", df)

    #Change dataframe to csv and save it:
    df.to_csv(os.path.join('../Outputs/Tables', 'Metric_based_Heuristics_Table_7.csv'))

    #Show result
    index_2 = ['Our study', 'Fernandes et al. (2016)']
    data_2 = {'God Class-Average precision%': [], 'God Class-Average recall%': [], 'Long Method-Average precision%': [],
              'Long Method-Average recall%': []}

    df_2 = result_2(df_god, df_method, index_2, data_2)
    print("Table 8: \n", df_2)

    #Change dataframe to csv format and save it.

    df_2.to_csv(os.path.join('../Outputs/Tables', 'Metric_based_Heuristics_Table_8.csv'))

main()



