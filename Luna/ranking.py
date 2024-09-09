import pandas as pd

def rank_pred(model, features): # features input refers to resultant df after calling get_features function, names input refers to corresponding names after calling get_names 
    probs = model.predict_proba(features)[:, 1] 

    df_probs = pd.DataFrame(features.copy())
    df_probs['Prediction Prob'] = probs

    df_sorted = df_probs.sort_values(by='Prediction Prob', ascending=False)

    columns = df_sorted.columns.tolist()
    new_order = columns[-1:] + columns[:-1] 
    df_sorted = df_sorted[new_order]

    #df_merged = names.join(df_sorted, how='inner')  # join on the row index
    #df_merged = df_merged.reindex(df_sorted.index)
    return df_sorted


def custom_rank_pred(model, features, criteria): # features input refers to resultant df after calling get_features function, names input refers to corresponding names after calling get_names 
    probs = model.predict_proba(features)[:, 1] 

    df_probs = pd.DataFrame(features.copy())
    df_probs['Prediction Prob'] = probs

    df_sorted = df_probs.sort_values(by=criteria, ascending=False)

    columns = df_sorted.columns.tolist()
    new_order = columns[-1:] + columns[:-1] 
    df_sorted = df_sorted[new_order]

    #df_merged = names.join(df_sorted, how='inner')  # join on the row index
    #df_merged = df_merged.reindex(df_sorted.index)
    return df_sorted

def get_names(df): # df refers to exensio dataset with test parameter names (before feature processing)
    return df.loc['Name'] 


