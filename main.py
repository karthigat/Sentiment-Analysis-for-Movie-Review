import csv # we need import file
import pandas as pd 

sentimet = pd.read_csv(r'prepd_data.csv', sep=",")
#sentimet.encode('utf-8').strip()
print(sentimet['review'])
