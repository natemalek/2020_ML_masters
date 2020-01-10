import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def basic_regression_model(train_file, test_file):
    '''
    Trains a logreg model on training data, predicts on test data,
    and returns the resulting r2_score.
    
    :param train_file: path to a pkl file with training data. This file
        should contain a pandas df structure with two columns: "Embedding"
        and "Valence score".
    :param test_file: path to a pkl file with test data, structured in the
        same way as train_file.
        
    :returns r2_score: a float. 
    '''
    df_train = pd.read_pickle(train_file)
    df_test = pd.read_pickle(test_file)
    
    X_train = list(df_train['Embedding'].values)
    y_train = df_train['Valence score'].values

    X_test = list(df_test['Embedding'].values)
    y_test = df_test['Valence score'].values
    
    linRegr = LinearRegression()
    model = linRegr.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    r_score = r2_score(y_test,predictions)
    
    return r_score

if __name__ == "__main__":
    
    train_filepath = 'data/embeddings_training.pkl'
    dev_filepath = 'data/embeddings_dev.pkl'
    test_filepath = 'data/embeddings_test.pkl'
    
    r_score = basic_regression_model(train_filepath, dev_filepath)
    print(f"basic model r2 score: {r_score}")
    