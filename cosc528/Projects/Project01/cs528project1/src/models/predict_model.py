import pandas as pd


DATADIR =

column_names = [
    'mpg', 'cylinders', 'displacement',
    'horsepower', 'weight', 'acceleration',
    'model year', 'origin', 'car name'
]
df = pd.read_csv('data/auto-mpg.data', header=None,
                 names=column_names, na_values='?',
                 sep='\s+')
