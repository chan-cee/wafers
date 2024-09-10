import pandas as pd
import numpy as np
import pickle
from scipy.stats import skew, kurtosis, norm, kstest, zscore, shapiro
from sklearn.neighbors import LocalOutlierFactor
import warnings
import os
import sys
from sklearn.preprocessing import StandardScaler
 
#IMPORT PICKLED SVM MODEL
file_path = r"V:\TimofeyDakh\BlunderbussV2\luna_only_model.pkl"
with open(file_path,'rb') as file:
	loaded_model = pickle.load(file)
 
directory_path = r"V:\TimofeyDakh\BlunderbussV2"
sys.path.append(directory_path)
from features import exensio_get_features, classify_qq_outliers
from ranking import rank_pred
#LOAD ULT FROM EY AS DATAFRAME
ULT_df = pd.DataFrame(data=ULT_raw_singlereal)
#PST_df = pd.DataFrame(data=PST)
ULT_split_column_df = pd.DataFrame(data=ULT_split_column)
#create new dataframe containining "TestName" and "Label"
#output_df = pd.DataFrame(columns=["Cu seed/ECP","TestName", "Outlier/Longtail?"])
 
 
#Delete fail dies from ULT_df
#ULT_df = ULT_df.drop(ULT_df[ULT_df['BinState'] == 'F'].index)
 
 
#MAKE LIST OF ALL ULT DATAFRAME COLUMN HEADERS
#PST_df1 = PST_df.drop(PST_df[PST_df['Parameter Unit'] == 'fct'].index)
#header_list = PST_df['Name'].tolist()
header_list = ULT_df.columns.tolist()
merged_ULT_df = pd.concat([ULT_df, ULT_split_column_df['Cu seed/ECP']], axis=1)
 
#MAKE LIST OF ALL UNIQUE WAFER IDs
unique_splits = merged_ULT_df["Cu seed/ECP"].unique()
unique_splits_list = unique_splits.tolist()
#LOOP THROUGH ULT COLUMN BY COLUMN, CALCULATE INPUTE FEATURE VECTOR FOR EACH COLUMN, THEN APPLY MODEL TO GENERATE LABEL AND OUTPUT TO OUTPUT DATAFRAME
 
#ranking_df = pd.DataFrame(columns=unique_splits_list)
#ranking_df = pd.DataFrame(index=header_list)
output_df = pd.DataFrame({"Test Name": header_list})

for split in unique_splits_list:
    single_split_df = merged_ULT_df[merged_ULT_df["Cu seed/ECP"] == split]

    #delete Cu seed column
    single_split_df = single_split_df.drop('Cu seed/ECP', axis=1)
    
    input_features = exensio_get_features(single_split_df)
    predictions = loaded_model.predict(input_features)
    confidences = rank_pred(loaded_model, input_features)

    output_df[f"Split {split} Outlier/Longtail?"] = predictions #pd.Series(predictions, index=header_list)
    output_df[f"Split {split} Confidence"] = confidences

    # for series_value, series2_value, list_value in zip(predictions, strengths, header_list):
    #     result = {"Cu seed/ECP": split, "TestName": list_value, "Outlier/Longtail?": series_value, "Strength_ranking": series2_value}
    
 
#Asssign df to output parameter to feed back into EY
#statistics_feature_model_output_48_features = pd.DataFrame().assign(TestName=output_df['TestName'], Label=output_df['Outlier/Longtail?'])

prediction_column_names = [col for col in output_df.columns if 'Outlier' in col]
output_df['All Splits Flag'] = (output_df[prediction_column_names] == 1).any(axis=1).astype(int)

#classification_by_splits = output_df
 
#LOAD ULT FROM EY AS DATAFRAME
#testitem_based_on_splits = output_df.groupby('Test Name')['Outlier/Longtail?'].apply(lambda x: 'yes' if 1 in x.values else 'no').reset_index()