import pandas as pd
import features
#import pickle

# with open(file_path,'rb') as file:
# 	model = pickle.load(file)

# ey_test_columns -> ULT_raw_singlereal; ey_split_column -> ULT_split_column; model -> with open(file_path,'rb') as file: model = pickle.load(file)
def get_predictions_ranking(ey_test_columns, ey_split_column, model):

    #LOAD ULT FROM EY AS DATAFRAME
    ULT_df = pd.DataFrame(data=ey_test_columns)
    ULT_split_column_df = pd.DataFrame(data=ey_split_column)
    merged_ULT_df = pd.concat([ULT_df, ULT_split_column_df['Cu seed/ECP']], axis=1)

    test_names = ULT_df.columns.tolist()
    unique_splits_list = merged_ULT_df["Cu seed/ECP"].unique().tolist()

    #ranking_df = pd.DataFrame(index=test_list)
    output_df = pd.DataFrame({"Test Name": test_names})

    for split in unique_splits_list:
        single_split_df = merged_ULT_df[merged_ULT_df["Cu seed/ECP"] == split]
        single_split_df = single_split_df.drop('Cu seed/ECP', axis=1)
        
        input_features = features.exensio_get_features(single_split_df)
        predictions = model.predict(input_features)
        confidences = rank_pred(model, input_features)

        output_df[f"Split {split} Outlier/Longtail?"] = predictions #pd.Series(predictions, index=header_list)
        output_df[f"Split {split} Confidence"] = confidences
        
    prediction_column_names = [col for col in output_df.columns if 'Outlier' in col]
    output_df['All Splits Flag'] = (output_df[prediction_column_names] == 1).any(axis=1).astype(int)

    return output_df



def rank_pred(model, features): # features input refers to resultant df after calling get_features function, names input refers to corresponding names after calling get_names 
    probs = model.predict_proba(features)[:, 1] 
    return probs

    # df_probs = pd.DataFrame(features.copy())
    # df_probs['Prediction Prob'] = probs
    
    # df_sorted = df_probs.sort_values(by='Prediction Prob', ascending=False)

    # return df_probs['Prediction Prob']

    # columns = df_sorted.columns.tolist()
    # new_order = columns[-1:] + columns[:-1] 
    # df_sorted = df_sorted[new_order]

    # #df_merged = names.join(df_sorted, how='inner')  # join on the row index
    # #df_merged = df_merged.reindex(df_sorted.index)
    # return df_sorted


def custom_rank_pred(model, features, criteria): # features input refers to resultant df after calling get_features function, names input refers to corresponding names after calling get_names 
    probs = model.predict_proba(features)[:, 1] 

    df_probs = pd.DataFrame(features.copy())
    df_probs['Prediction Prob'] = probs

    df_sorted = df_probs.sort_values(by=criteria, ascending=False)

    # columns = df_sorted.columns.tolist()
    # new_order = columns[-1:] + columns[:-1] 
    # df_sorted = df_sorted[new_order]

    #df_merged = names.join(df_sorted, how='inner')  # join on the row index
    #df_merged = df_merged.reindex(df_sorted.index)
    return df_probs['Prediction Prob']


# def get_names(df): # df refers to exensio dataset with test parameter names (before feature processing)
#     return df.loc['Name'] 


