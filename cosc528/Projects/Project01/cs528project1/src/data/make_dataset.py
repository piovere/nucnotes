# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os.path
import pandas as pd
import numpy as np
from math import floor


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    rootdir = Path(__file__).resolve().parents[2]
    datafile = os.path.join(rootdir, 'data', 'raw', 'auto-mpg.data')

    column_names = [
        'mpg', 'cylinders', 'displacement',
        'horsepower', 'weight', 'acceleration',
        'model year', 'origin', 'car name'
    ]
    df = pd.read_csv('data/auto-mpg.data', header=None,
                     names=column_names, na_values='?',
                     sep='\s+')

    # Scrub NA's
    df = df.dropna(axis=0)

    # Targets
    y = df['mpg']
    df.drop(['mpg'])

    # Add the inverted columns
    df['inverse displacement'] = df['displacement'] ** -1
    df['inverse horsepower'] = df['horsepower'] ** -1
    df['inverse weight'] = df['weight'] ** -1

    # Split into train, test, val
    xtr, ytr, xts, yts, xv, yv = train_test_val_split(df)

    processed_dir = os.path.join(rootdir, 'data', 'processed')
    train_file = df.to_csv()


def train_test_val_split(x, y, train_frac=0.5, test_frac=0.25, val_frac=0.25, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    else:
        np.random.seed()

    # Normalize fractions
    tot = train_frac + test_frac + val_frac
    train_frac /= tot
    test_frac /= tot
    val_frac /= tot

    data = x

    numrows = data.shape[0]

    rows = np.arange(numrows)
    rows = np.random.shuffle(rows)

    train_row_count = floor(train_frac * numrows)
    test_row_count = floor(test_frac * numrows)
    val_row_count = floor(val_frac * numrows)

    train_test_boundary = train_row_count
    test_val_boundary = train_test_boundary + test_row_count
    val_train_boundary = test_val_boundary + val_row_count

    train_rows = rows[:train_test_boundary]
    test_rows = rows[train_test_boundary:test_val_boundary]
    val_rows = rows[test_val_boundary:val_train_boundary]

    # Add leftover rows to training
    np.append(train_rows, rows[val_train_boundary:])

    train_x = data.loc[train_rows]
    test_x = data.loc[test_rows]
    val_x = data.loc[val_rows]

    # Select y values at same indices
    train_y = y.loc[train_rows]
    test_y = y.loc[test_rows]
    val_y = y.loc[val_rows]

    return train_x, train_y, test_x, test_y, val_x, val_y


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
