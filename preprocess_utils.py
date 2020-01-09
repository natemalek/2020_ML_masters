# Utils for preprocessing
import pandas as pd

def import_file(filepath):
    '''
    Takes a filepath to a headerful tsv file and returns the data as a
    pandas df.
    '''
    
    df = pd.read_csv(filepath, delimiter="\t")

    return df
