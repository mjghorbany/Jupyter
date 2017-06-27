import csv, sys,re
import nltk
import pandas as pd

print('Hella \n')

filename = 'f3.csv'

with open(filename, 'r') as f:
    reader = csv.reader(f)
    csvlist = list(reader)

'''
for df in pd.read_csv(filename,sep=',', header = None, chunksize=1):
    print(df)
    jsondf=df.to_json()
    print(jsondf)
'''

def lexical_diversity(text):
    return len(set(text)) / len(text)

df2 = pd.read_csv(filename)
print('Next \n')
#print(df2.loc[lambda df: df.Qty > 0, :])
#print(df2[lambda df: df.columns[3]])
#print(df2[:3])

#print(df2.dtypes)

#titles=df2.columns
#print(titles)

#print(df2.dtypes)
print(df2)

'''
with open(filename, 'r') as f:
    reader = csv.reader(f)
    csvlist=reader
    try:
        for row in reader:
            print(row)
            #print(len(row))
    except csv.Error as e:
        sys.exit('file %s, line %d: %s' % (filename, reader.line_num, e))


print (csvlist)
str1 = ''.join(str(e) for e in csvlist)
'''


#print(csvlist[4])
print('\nThe End ')



