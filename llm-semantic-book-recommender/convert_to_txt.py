import pandas as pd

df = pd.read_csv('data/tagged_description.csv')
with open('data/tagged_description.txt', 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        f.write(f'{row["isbn13"]} {row["description"]}\n') 