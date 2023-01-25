# Table 11
# Comparing the F-measure of metric-based classifiers on the MLCQ dataset (our expriment) to performances reported in the original studies
import pandas as pd
dc = pd.read_csv('../Outputs/Tables/God_Class_Table9.csv')
de = pd.read_csv('../Outputs/Tables/long_method_Table10.csv')
hgc = dc.loc[2].at["F-measure"]
mlgc = dc.loc[3].at["F-measure"]
hlm = de.loc[1].at["F-measure"]
mllm = de.loc[4].at["F-measure"]

df = pd.DataFrame(columns=['Proposed approach', 'Dataset', 'Original study'])
result = f'ML approach:  GC:{mlgc}   LM:{mllm} ,  Heuristics:  GC:{hgc}   LM:{hlm}'
data = {'Proposed approach':['Fontana et al. (2016) : The authors used many code metrics as featgures (61 class level and 82 method level metrics). The authors also experimented with a wide range of ML classifiers.Here, we report the best performance. Experiment: 10-fold-cross validation', 'DiNucci et al.(2018): The autors replicate the study by Fontana et al.(2016) but correct the identified limitations.', 'Pecorelli et al. (2020) :This study compares the performance of ML-based and heuristic detectors. In the ML based approach, authors use 17 code metrics as features. These were the same metrics used in heuristics.The authors provide an online appendix for experiment replication. However, the appendix contains the extracted metrics without the source code needed to run our experiments.Though the dataset consists of open-source projects, we lacked the details necessary for reliable matching with the source code.Experiment: 10-fold-crossvalidation.' ,'Aleksandar et al(2022):This study compares the performance of multiple ML-based code smell detection models against multiple metric-based heuristics for detection of God Class and Long Method code smells.'],
        'Dataset':['Qualitas corpus – 420 instances per smell sampled from 74 open-source systems. Labeling procedure: semiautomatic. Cross-checked labels: yes. Annotator training: yes.The authors enforced anunrealistic ratio of code smells to non-smells on both the training and the test set.', 'The authors use the same dataset as Fontana et al.(2016) but correct the artificial smell to the non smell ratio on the wholedataset.', 'Palomba et al. (2018a) – 125 releases of 13 open-source projects. Labeling procedure: semiautomatic. Annotator training: no. Cross-checked labels: yes.', 'MLCQ dataset with nearly 15000 code samples was created by sofware developers with professional exprience who reviewed industry-relevant, contemporary Java Open source projects. Labeling procedure: Manually'],
       'Original study':['ML approach: GC:0.98 LM:1.0', 'ML approach:GC:0.5 LM:0.48', 'ML approach: GC:0.41 LM:0.23 , Heuristics: GC:0.16 LM:0.44' , result]}
Comparing_MLCQ_Resualt_Table11 = pd.DataFrame(data)
Comparing_MLCQ_Resualt_Table11.to_csv('../Outputs/Tables/Comparing_MLCQ_Result_Table11.csv')
print(Comparing_MLCQ_Resualt_Table11)
