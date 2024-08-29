
import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, norm, kstest, zscore, shapiro
from sklearn.neighbors import LocalOutlierFactor
import warnings

from sklearn.preprocessing import StandardScaler

# input is df where each test (parameter) is represented by a column 
def exensio_get_features(df):
    feature_keys = [
        'Mean', 'Median', 'Std_Dev', 'IQR', 'Skewness', 'Kurtosis', 'Min', 'Max', 
        'Range', 'Upper_Tail', 'Lower_Tail', 'Extreme_Tail_95', 'Extreme_Tail_99', 
        'Extreme_Tail_05', 'Extreme_Tail_01', 'Upper_Tail_Mean', 'Upper_Tail_Var', 
        'Lower_Tail_Mean', 'Lower_Tail_Var', 'Tail_Weight_Ratio', 'Excess_Kurtosis', 
        'P99', 'P1', 'Outliers_Zscore', 'Outliers_IQR', 'KS_Stat_norm', 'KS_P_value_norm', 
        'Shapiro_Stat', 'Shapiro_P_value'
    ]
    feature_vectors = []    
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(normalized_data) 
    for column in df_scaled.columns:  #iterate through each test  
        values = df_scaled[column].dropna()
        if values.empty:
            empty_vector = {key: [0] for key in feature_keys}
            feature_vectors.append(empty_vector)
            continue

        unique_values = np.unique(values)
        num_unique = len(unique_values)

        # statistical properties
        count = len(values)
        mean = values.mean()
        median = values.median()
        std_dev = values.std()
        iqr = values.quantile(0.75) - values.quantile(0.25)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            try:
                if num_unique <= 2:
                    skewness = 0
                    kurt = 0
                else:
                    skewness = skew(values)
                    kurt = kurtosis(values)
            except RuntimeWarning:
                skewness = np.nan
                kurt = np.nan
        min_value = values.min()
        max_value = values.max()
        range = max_value - min_value
        p95 = values.quantile(0.95)
        p5 = values.quantile(0.05)
        p99 = values.quantile(0.99)
        p1 = values.quantile(0.01)
        upper_tail_diff = p99 - p95
        lower_tail_diff = p5 - p1
        tail_length_95 = max_value - p95
        tail_length_99 = max_value - p99
        tail_length_05 = p5 - min_value 
        tail_length_01 = p1 - min_value
        upper_tail = values[values >= values.quantile(0.95)]
        lower_tail = values[values <= values.quantile(0.05)]
        if num_unique <= 2: # discrete distribution?
            upper_tail_mean = 0
            upper_tail_var = 0
            lower_tail_mean = 0
            lower_tail_var = 0
        else: 
            upper_tail_mean = upper_tail.mean()
            upper_tail_var = upper_tail.var()
            lower_tail_mean = lower_tail.mean()
            lower_tail_var = lower_tail.var()
        #percentile_ratio = p95 / p5 if p5 != 0 else np.nan
        tail_weight_ratio = np.sum((values > (median + 1.5*iqr)) | (values < (median - 1.5*iqr))) / count
        tail_length_ratio_95 = tail_length_95 / range if iqr > 0 else 0
        tail_length_ratio_05 = tail_length_05 / range if iqr > 0 else 0
        excess_kurtosis = kurt - 3

        # outlier detection
        zscores = zscore(values)
        outliers_zscore = np.abs(zscores) > 3 #zscores.abs() > 3 
        count_outliers_zscore = outliers_zscore.sum() # using zscore
        count_outliers_zscore_prop = count_outliers_zscore / count

        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1   
        outliers_iqr = ((values < (Q1 - 1.5 * IQR)) | values > (Q3 + 1.5 * IQR)) # using iqr
        count_outliers_iqr = outliers_iqr.sum()
        count_outliers_iqr_prop = count_outliers_iqr / count

        outliers_tukey = (values > (Q3 + 3 * IQR)) | (values < (Q1 - 3 * IQR))
        count_outliers_tukey = outliers_tukey.sum()
        count_outliers_tukey_prop = count_outliers_tukey / count

        outliers_qq, long_tail_qq = classify_qq_outliers(values)
        count_positive_qq = outliers_qq + long_tail_qq

        # goodness of fit parameters
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fitted_mean, fitted_std_dev = norm.fit(values)
        warnings.filterwarnings("ignore", message="scipy.stats.shapiro: For N > 5000")

        if num_unique <= 2: 
            ks_stat_norm, ks_p_value_norm = 1, 0
            shapiro_stat, shapiro_p_value = 0, 0
        else:
            ks_stat_norm, ks_p_value_norm = kstest(values, 'norm', args=(fitted_mean, fitted_std_dev))
            shapiro_stat, shapiro_p_value = shapiro(values)

        # ks_stat_norm, ks_p_value_norm = kstest(values, 'norm', args=(fitted_mean, fitted_std_dev))

        # if len(values) >= 3 and np.ptp(values) > 0:
        #     shapiro_stat, shapiro_p_value = shapiro(values)
        # else:
        #     shapiro_stat, shapiro_p_value = np.nan, np.nan #float('nan'), float('nan')

        feature_vector = {
                'Count': count,
                'Unique_Count': num_unique,
                'Mean': mean,
                'Median': median,
                'Std_Dev': std_dev,
                'IQR': iqr,
                'Skewness': skewness,
                'Kurtosis': kurt,
                'Min': min_value,
                'Max': max_value,
                'Range': range,
                'Upper_Tail': upper_tail_diff,
                'Lower_Tail': lower_tail_diff,
                'Extreme_Tail_95': tail_length_95,
                'Extreme_Tail_99': tail_length_99,
                'Extreme_Tail_05': tail_length_05,
                'Extreme_Tail_01': tail_length_01,
                'Upper_Tail_Mean': upper_tail_mean,
                'Upper_Tail_Var': upper_tail_var,
                'Lower_Tail_Mean': lower_tail_mean,
                'Lower_Tail_Mean': lower_tail_var,
                #'Percentile_Ratio_95_5': percentile_ratio,
                'Tail_Weight_Ratio': tail_weight_ratio,
                'Tail_Length_Ratio_95': tail_length_ratio_95,
                'Tail_Length_Ratio_05': tail_length_ratio_05,
                'Excess_Kurtosis': excess_kurtosis,
                'P99': p99,
                'P1': p1,
                'Outliers_Zscore': count_outliers_zscore,
                'Outliers_Zscore_prop': count_outliers_zscore_prop,
                'Outliers_IQR': count_outliers_iqr,
                'Outliers_IQR_prop': count_outliers_iqr_prop,
                'Outliers_Tukey': count_outliers_tukey,
                'Outliers_Tukey_prop': count_outliers_tukey_prop,
                'QQ Count': count_positive_qq,
                'KS_Stat_norm': ks_stat_norm,
                'KS_P_value_norm': ks_p_value_norm,
                'Shapiro_Stat': shapiro_stat,
                'Shapiro_P_value': shapiro_p_value
        }
        feature_vectors.append(feature_vector)

    features_df = pd.DataFrame(feature_vectors)
    return features_df


from sklearn.cluster import DBSCAN
import warnings
# for my ipynb pre-processing
def get_features(df):
    feature_vectors = []
    for column in df.columns:        
        values = df[column].dropna()
        # if values.empty:
        #     return pd.DataFrame() # return empty features if no values 
        unique_values = np.unique(values)
        num_unique = len(unique_values)

        # if num_unique < 3:
        #     is_functional = 1
        #     # assign everything 0? create new variable?

        # statistical properties
        count = len(values)
        mean = values.mean()
        median = values.median()
        std_dev = values.std()
        iqr = values.quantile(0.75) - values.quantile(0.25)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            try:
                if num_unique <= 2:
                    skewness = 0
                    kurt = 0
                else:
                    skewness = skew(values)
                    kurt = kurtosis(values)
            except RuntimeWarning:
                skewness = np.nan
                kurt = np.nan
        min_value = values.min()
        max_value = values.max()
        range = max_value - min_value
        p95 = values.quantile(0.95)
        p5 = values.quantile(0.05)
        p99 = values.quantile(0.99)
        p1 = values.quantile(0.01)
        upper_tail_diff = p99 - p95
        lower_tail_diff = p5 - p1
        tail_length_95 = max_value - p95
        tail_length_99 = max_value - p99
        tail_length_05 = p5 - min_value 
        tail_length_01 = p1 - min_value
        upper_tail = values[values >= values.quantile(0.95)]
        lower_tail = values[values <= values.quantile(0.05)]
        if num_unique <= 2: # discrete distribution?
            upper_tail_mean = 0
            upper_tail_var = 0
            lower_tail_mean = 0
            lower_tail_var = 0
        else: 
            upper_tail_mean = upper_tail.mean()
            upper_tail_var = upper_tail.var()
            lower_tail_mean = lower_tail.mean()
            lower_tail_var = lower_tail.var()
        #percentile_ratio = p95 / p5 if p5 != 0 else np.nan
        tail_weight_ratio = np.sum((values > (median + 1.5*iqr)) | (values < (median - 1.5*iqr))) / count
        tail_length_ratio_95 = tail_length_95 / range if iqr > 0 else 0
        tail_length_ratio_05 = tail_length_05 / range if iqr > 0 else 0
        excess_kurtosis = kurt - 3

        # outlier detection
        zscores = zscore(values)
        outliers_zscore = np.abs(zscores) > 3 #zscores.abs() > 3 
        count_outliers_zscore = outliers_zscore.sum() # using zscore
        count_outliers_zscore_prop = count_outliers_zscore / count

        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1   
        outliers_iqr = ((values < (Q1 - 1.5 * IQR)) | values > (Q3 + 1.5 * IQR)) # using iqr
        count_outliers_iqr = outliers_iqr.sum()
        count_outliers_iqr_prop = count_outliers_iqr / count

        outliers_tukey = (values > (Q3 + 3 * IQR)) | (values < (Q1 - 3 * IQR))
        count_outliers_tukey = outliers_tukey.sum()
        count_outliers_tukey_prop = count_outliers_tukey / count

        outliers_qq, long_tail_qq = classify_qq_outliers(values)
        count_positive_qq = outliers_qq + long_tail_qq

        # goodness of fit parameters
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fitted_mean, fitted_std_dev = norm.fit(values)
        warnings.filterwarnings("ignore", message="scipy.stats.shapiro: For N > 5000")

        if num_unique <= 2: 
            ks_stat_norm, ks_p_value_norm = 1, 0
            shapiro_stat, shapiro_p_value = 0, 0
        else:
            ks_stat_norm, ks_p_value_norm = kstest(values, 'norm', args=(fitted_mean, fitted_std_dev))
            shapiro_stat, shapiro_p_value = shapiro(values)

        # ks_stat_norm, ks_p_value_norm = kstest(values, 'norm', args=(fitted_mean, fitted_std_dev))

        # if len(values) >= 3 and np.ptp(values) > 0:
        #     shapiro_stat, shapiro_p_value = shapiro(values)
        # else:
        #     shapiro_stat, shapiro_p_value = np.nan, np.nan #float('nan'), float('nan')

        feature_vector = {
                'Count': count,
                'Unique_Count': num_unique,
                'Mean': mean,
                'Median': median,
                'Std_Dev': std_dev,
                'IQR': iqr,
                'Skewness': skewness,
                'Kurtosis': kurt,
                'Min': min_value,
                'Max': max_value,
                'Range': range,
                'Upper_Tail': upper_tail_diff,
                'Lower_Tail': lower_tail_diff,
                'Extreme_Tail_95': tail_length_95,
                'Extreme_Tail_99': tail_length_99,
                'Extreme_Tail_05': tail_length_05,
                'Extreme_Tail_01': tail_length_01,
                'Upper_Tail_Mean': upper_tail_mean,
                'Upper_Tail_Var': upper_tail_var,
                'Lower_Tail_Mean': lower_tail_mean,
                'Lower_Tail_Mean': lower_tail_var,
                #'Percentile_Ratio_95_5': percentile_ratio,
                'Tail_Weight_Ratio': tail_weight_ratio,
                'Tail_Length_Ratio_95': tail_length_ratio_95,
                'Tail_Length_Ratio_05': tail_length_ratio_05,
                'Excess_Kurtosis': excess_kurtosis,
                'P99': p99,
                'P1': p1,
                'Outliers_Zscore': count_outliers_zscore,
                'Outliers_Zscore_prop': count_outliers_zscore_prop,
                'Outliers_IQR': count_outliers_iqr,
                'Outliers_IQR_prop': count_outliers_iqr_prop,
                'Outliers_Tukey': count_outliers_tukey,
                'Outliers_Tukey_prop': count_outliers_tukey_prop,
                'QQ Count': count_positive_qq,
                'KS_Stat_norm': ks_stat_norm,
                'KS_P_value_norm': ks_p_value_norm,
                'Shapiro_Stat': shapiro_stat,
                'Shapiro_P_value': shapiro_p_value
        }
        feature_vectors.append(feature_vector)

    features_df = pd.DataFrame(feature_vectors)
    return features_df


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def classify_qq_outliers(data, outlier_threshold=4, long_tail_threshold=4):
    
    data = pd.to_numeric(data , errors='coerce').dropna()

    if np.std(data) < 1e-8: # data is nearly constant
        return 0, 0

    # comparing to theoretical distribution
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm", plot=None)
    
    # calculate deviation from the theoretical line
    theoretical_line = slope * osm + intercept
    deviations = osr - theoretical_line
    
    # identify outliers: points that deviate significantly from the line
    outliers = np.abs(deviations) > outlier_threshold * np.std(deviations)
    
    # identify long-tail behavior: consistent deviation in the tails
    long_tail_upper = deviations > long_tail_threshold * np.std(deviations)
    long_tail_lower = deviations < -long_tail_threshold * np.std(deviations)
    long_tail = long_tail_upper | long_tail_lower

    both_outlier_and_long_tail = outliers & long_tail

    outliers_count = np.sum(outliers)
    longtail_count = np.sum(long_tail)

    return outliers_count, longtail_count

# Sub function to engineer statistical features for each test column
def get_single_test_features(test): 
    values = test.dropna()
    if values.empty:
        return pd.DataFrame() # return empty features if no values 
    # statistical properties
    mean = values.mean()
    median = values.median()
    std_dev = values.std()
    iqr = values.quantile(0.75) - values.quantile(0.25)
    unique_values = np.unique(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            if len(unique_values) <= 2:
                skewness = 0
                kurt = 0
            else:
                skewness = skew(values)
                kurt = kurtosis(values)
        except RuntimeWarning:
            skewness = np.nan
            kurt = np.nan
    min_value = values.min()
    max_value = values.max()
    range = max_value - min_value
    p95 = values.quantile(0.95)
    p5 = values.quantile(0.05)
    p99 = values.quantile(0.99)
    p1 = values.quantile(0.01)
    upper_tail_diff = p99 - p95
    lower_tail_diff = p5 - p1
    tail_length_95 = max_value - p95
    tail_length_99 = max_value - p99
    tail_length_05 = p5 - min_value 
    tail_length_01 = p1 - min_value 
    #percentile_ratio = p95 / p5 if p5 != 0 else np.nan
    tail_weight_ratio = np.sum((values > (median + 1.5*iqr)) | (values < (median - 1.5*iqr))) / len(values)
    excess_kurtosis = kurt - 3
    p99 = values.quantile(0.99)
    p1 = values.quantile(0.01)

    # outlier detection
    zscores = zscore(values)
    outliers_zscore = zscores.abs() > 3 
    count_outliers_zscore = outliers_zscore.sum() # using zscore
    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1   
    outliers_iqr = ((values < (Q1 - 1.5 * IQR)) | values > (Q3 + 1.5 * IQR)) # using iqr
    count_outliers_iqr = outliers_iqr.sum()
    if len(values) > 1:
        lof = LocalOutlierFactor(n_neighbors=min(20, len(values) - 1)) # using local density of points
        lof_scores = lof.fit_predict(values.values.reshape(-1, 1))
        count_outliers_lof = np.sum(lof_scores == -1)
    else:
        count_outliers_lof = 0
    # goodness of fit parameters
    fitted_mean, fitted_std_dev = norm.fit(values)
    ks_stat_norm, ks_p_value_norm = kstest(values, 'norm', args=(fitted_mean, fitted_std_dev))
    warnings.filterwarnings("ignore", message="scipy.stats.shapiro: For N > 5000")
    if len(values) >= 3:
        shapiro_stat, shapiro_p_value = shapiro(values)
    else:
        shapiro_stat, shapiro_p_value = np.nan, np.nan #float('nan'), float('nan')
        print('nan')

    feature_vector = {
            'Mean': mean,
            'Median': median,
            'Std_Dev': std_dev,
            'IQR': iqr,
            'Skewness': skewness,
            'Kurtosis': kurt,
            'Min': min_value,
            'Max': max_value,
            'Range': range,
            'Upper_Tail': upper_tail_diff,
            'Lower_Tail': lower_tail_diff,
            'Extreme_Tail_95': tail_length_95,
            'Extreme_Tail_99': tail_length_99,
            'Extreme_Tail_05': tail_length_05,
            'Extreme_Tail_01': tail_length_01,
            #'Percentile_Ratio_95_5': percentile_ratio,
            'Tail_Weight_Ratio': tail_weight_ratio,
            'Excess_Kurtosis': excess_kurtosis,
            'P99': p99,
            'P1': p1,
            'Outliers_Zscore': count_outliers_zscore,
            'Outliers_IQR': count_outliers_iqr,
            'Outliers_LOF': count_outliers_lof,
            'KS_Stat_norm': ks_stat_norm,
            'KS_P_value_norm': ks_p_value_norm,
            'Shapiro_Stat': shapiro_stat,
            'Shapiro_P_value': shapiro_p_value
    }

    features = pd.DataFrame([feature_vector.values()], columns=feature_vector.keys())
    return features



# Overall function to aggregate chosen test columns into feature dataframe
# input consists of columns which are filtered to only include proper parameter/tests with numerical values for classifying distribution  
# output consists of a dataframe of features, where each row represent the features corresponding to each test

from sklearn.preprocessing import StandardScaler
import pandas as pd

# input consists of columns which are filtered to only include proper parameter/tests with numerical values for classifying distribution
def get_all_test_features(df):
    feature_list = []
    scaler = StandardScaler()
    for test_name in df.columns:
        test_data = df[test_name].values.reshape(-1, 1)
        test_transformed = scaler.fit_transform(test_data) 
        test_transformed = test_transformed.flatten()
        features = get_single_test_features(pd.Series(test_transformed))
        # features['Test'] = test_name 
        feature_list.append(features)

    result = pd.concat(feature_list, axis=0).reset_index(drop=True)
    return result


if __name__ == "__main__":
    YK_data = pd.read_csv('../../data/YK_training_ULT.csv')

    column = YK_data.iloc[:,112]
    feature = get_single_test_features(column)
    print(feature)

    columns = YK_data.iloc[:,112:115]
    features = get_all_test_features(columns)
    print(features)

