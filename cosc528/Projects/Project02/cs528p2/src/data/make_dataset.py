# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    df = pd.read_excel(input_filepath, na_values=["-"],
                   nrows=57, true_values=["x"])
    # Fill in missing values from research
    # Add Clemson IPEDS number
    ix = np.where(df["Name"] == "Clemson Univ.")[0][0]
    df.at[ix, "IPEDS#"] = 217882

    # Drop "HBC" since it has no variance
    # Drop "Med School Res $" since it's missing for many schools
    # Drop "AG Research ($000)" since it's missing for many schoools
    # Drop "Wall St. Jourl Rank" since it's missing for many schoools
    drop_columns = [
        "HBC", 
        "Med School Res $",
        "AG Research ($000)",
        "Wall St. Jourl Rank"
    ]
    df_clean = df.drop(drop_columns, 'columns')

    # Some of these columns are categorical.
    # Convert them to one-hot
    categorical_columns = [
        "Carm R1",
        "2014 Med School",
        "Vet School",
    ]
    df_clean = pd.get_dummies(df_clean, columns=categorical_columns)

    # Clemson endowment in 2017 was $608,000,000
    # Source: https://www.clemson.edu/giving/cufoundations/documents/allocations.pdf
    ix = np.where(df["Name"] == "Clemson Univ.")[0][0]
    df_clean.at[ix, "Endowment Figure"] = 608000000
    df_clean.at[ix, "Endowment"] = 608000000
    df_clean.at[ix, "Enowment / St. FTE"] = \
        df_clean.at[ix, "Endowment"] / df_clean.at[ix, "ST. FTE"]

    # Save to file
    logger.info(f'Writing processed data to {output_filepath}')
    df_clean.to_csv(output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
