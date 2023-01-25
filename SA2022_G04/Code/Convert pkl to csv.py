import pickle
import pandas as pd
import os

with open('../Data/God_Class_metrics_and_heuristics.pkl', 'rb') as f:
    data = pickle.load(f)

with open('../Data/Long_Method_CuBERT_embeddings.pkl', 'rb') as f:
    test = pickle.load(f)

print("data: \n", data)
print("test: \n", test)

df = pd.DataFrame(data)
df.to_csv(os.path.join('../Data',r'God_Class_metrics_and_heuristics.csv'))

df = pd.DataFrame(test)
df.to_csv(os.path.join('../Data',r'Long_Method_CuBERT_embeddings.csv'))

