import pandas as pd

df = pd.DataFrame({"Name": ["Braund, Mr. Owen Harris","Allen, Mr. William Henry","Bonnell, Miss. Elizabeth"],"Age": [22, 35, 58],"Sex": ["male", "male", "female"]})

df = pd.DataFrame({"Letter": ["A","B","C","D","E","F","G","H","I","J","K"],"Code": ["Alpha","Bravo","Charlie","Delta","Echo","Fhox","Golf","Hotel","India","Juliet","Kilo"]})

## target = df[['Sex']]

## target = df.loc[0, 'Name']

## target = df.iloc[0, 1]

target = df.ix[0, 1]

print(target)