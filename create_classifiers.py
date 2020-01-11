import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
#from sklearn.metrics import r2_score
from scipy.stats.stats import pearsonr
import xgboost as xgb
# xgboost tutoral used: https://www.datacamp.com/community/tutorials/xgboost-in-python

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
    
    #lr_score = r2_score(y_test,predictions)
    pearson_r, p_value = pearsonr(y_test, predictions)
    
    return pearson_r, p_value

def basic_xgboost_model(train_file, test_file):
    '''
    Trains an xgboost model on training data, predicts on test data,
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
    
    xg_reg = xgb.XGBRegressor(objective ='reg:linear', subsample=0.75, learning_rate = 0.1,
                max_depth = 5, n_estimators = 300)
    
    xg_reg.fit(X_train, y_train)
    predictions = xg_reg.predict(X_test)
    
    #xgb_score = r2_score(y_test,predictions)
    pearson_r, p_value = pearsonr(y_test, predictions)
    
    return pearson_r, p_value

if __name__ == "__main__":
    
    train_filepath = 'data/embeddings/embeddings_training.pkl'
    dev_filepath = 'data/embeddings/embeddings_dev.pkl'
    test_filepath = 'data/embeddings/embeddings_test.pkl'
    
    lr_score, lr_p_value = basic_regression_model(train_filepath, dev_filepath)
    lr_score_test, lr_p_value_test = basic_regression_model(train_filepath, test_filepath)
    xgb_score_dev, xbg_dev_p_value = basic_xgboost_model(train_filepath, dev_filepath)
    xgb_score_test, xgb_test_p_value = basic_xgboost_model(train_filepath, test_filepath)
    print(f"basic LR model pearson r, p_value: {lr_score, lr_p_value}")
    print(f"basic LR model pearson r, p_value: {lr_score_test, lr_p_value_test}")
    print(f"basic XGB model dev pearson r, p_value: {xgb_score_dev, xbg_dev_p_value}")
    print(f"basic XGB model test pearson r, p_value: {xgb_score_test, xgb_test_p_value}")
    