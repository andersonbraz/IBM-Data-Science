import pandas as pd

csv_path = 'file.csv'
xls_path = 'file.xlsx'

df = pd.read_csv(csv_path)

## print(df)

result = df.head()

print(result)