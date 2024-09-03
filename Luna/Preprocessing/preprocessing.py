import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import features

def retrieve_test_data(values_df, label_df): 
    unlabelled_tests = get_unlabelled_tests(label_df)
    valid_labels = drop_unlabelled(label_df, unlabelled_tests)

    tests_only = get_tests_only(values_df, valid_labels) # essentially drops corresponding test columns without label

    targets = valid_labels[['Name','Distribution Type']] # extract out relevant columns 

    tests_only_T = tests_only.T.reset_index(inplace=False, drop=False)
    tests_only_T.columns = ['Name'] + [i for i in range(1, len(tests_only_T.columns))]

    merged_df = pd.merge(targets, tests_only_T, on='Name', how='inner') # merge values and target labels (each row is a test)
    final_df = merged_df.T # each column is test

    return final_df

def retrieve_features(df):
    test_values = df.iloc[2:]
    labels = df.iloc[:2]

    scaler = MinMaxScaler()
    test_scaled = pd.DataFrame(scaler.fit_transform(test_values))

    features_set = features.get_features(test_scaled)
    final_df = pd.concat([labels.T, features_set], axis=1)  

    return final_df

# function to add new column 'Target' 
def labelling_features(df, function):
    df['Target'] = df['Distribution Type'].apply(function) # new column created 

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df['Target']) # new column created
    df['Target_Encoded'] = y_encoded

    reordered_columns = ['Name'] + ['Target'] + ['Target_Encoded'] + [col for col in df.columns if col not in ['Target', 'Name', 'Target_Encoded']]
    final_df = df[reordered_columns]        

    return final_df

def drop_null_tests(df):
    null_df = df[df.isnull().any(axis=1)]
    print("Number of rows dropped: ", null_df.shape[0])
    final_df = df.dropna()
    return final_df


# HELPER FUNCTIONS 

# Function to drop columns (tests) in the raw data file which are NOT tests (i.e metadata like start time, lot number etc)
def get_tests_only(datafile,labelfile):
    keys = list(labelfile['Name'])
    df = datafile[keys]
    return df

# Function to find out non applicable rows (no distribution, not part of training data)
def get_unlabelled_tests(df_label):
    null_rowsLabel = df_label[df_label['Distribution Type'].isnull()]
    null_rowsLabel = list(null_rowsLabel['Name'])
    return null_rowsLabel

def drop_unlabelled(label_df, not_applicable):
    df = label_df.drop(label_df[label_df.Name.isin(not_applicable)].index)
    return df

def relabel_luna(original_label):
    if original_label in ['outlier', 'longtail']:  
        return 'Outlier/Longtail'
    else:
        return 'Other'