import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, norm, kstest, zscore, shapiro
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def find_discrete(features): # input is processed feature df from get_features
    df = pd.DataFrame(features.copy()) 
    result = [] 
    # df['Prediction'] = preds
    for index, row in df.iterrows():
        # identify discrete using threshold of 10?
        if row['Unique_Count'] < 10:
            result.append(index)
    return result

def handle_discrete(index, values, preds): # given row number from features corresponds to the test value columns index
    discrete_preds = preds[index] # retrive prediction for discrete
    selected_columns = values.iloc[:, index]
    for test in selected_columns.columns: 
        test_values = selected_columns[test].dropna()
    


# input is df where each test (parameter) is represented by a column 
def exensio_get_features(df):
    feature_keys = ['Unique_Count', 'Mean', 'Median', 'Std_Dev', 'IQR', 'Skewness',
    'Kurtosis', 'Min', 'Max', 'Range', 'Upper_Tail', 'Lower_Tail', 'Extreme_Tail_95',
    'Extreme_Tail_99', 'Extreme_Tail_05', 'Extreme_Tail_01', 'Upper_Tail_Mean',
    'Upper_Tail_Var', 'Lower_Tail_Mean', 'Lower_Tail_Var', 'Tail_Weight_Ratio',
    'Tail_Length_Ratio_95', 'Tail_Length_Ratio_05', 'Excess_Kurtosis', 'P99', 'P1',
    'Outliers_Zscore', 'Outliers_Zscore_prop', 'Outliers_IQR', 'Outliers_IQR_prop',
    'Outliers_Tukey', 'Outliers_Tukey_prop', 'QQ Count', 'KS_Stat_norm', 'KS_P_value_norm',
    'Shapiro_Stat', 'Shapiro_P_value']
    
    scaler = MinMaxScaler()
    minmax_data = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(minmax_data, columns=df.columns)
    
    def compute_features(column):
        values = column.dropna().values
        if len(values) == 0:
            return {key: 0 for key in feature_keys}

        unique_values = np.unique(values)
        num_unique = len(unique_values)

        # Statistical properties
        count = len(values)
        mean = np.mean(values)
        median = np.median(values)
        std_dev = np.std(values)
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        skewness = skew(values) if num_unique > 2 else 0
        kurt = kurtosis(values) if num_unique > 2 else 0
        min_value = np.min(values)
        max_value = np.max(values)
        range_ = max_value - min_value
        p95, p5, p99, p1 = np.percentile(values, [95, 5, 99, 1])
        upper_tail_diff = p99 - p95
        lower_tail_diff = p5 - p1
        tail_length_95 = max_value - p95
        tail_length_99 = max_value - p99
        tail_length_05 = p5 - min_value
        tail_length_01 = p1 - min_value
        upper_tail = values[values >= p95]
        lower_tail = values[values <= p5]

        if num_unique <= 2: 
            upper_tail_mean = 0
            upper_tail_var = 0
            lower_tail_mean = 0
            lower_tail_var = 0
        else: 
            upper_tail_mean = np.mean(upper_tail)
            upper_tail_var = np.var(upper_tail)
            lower_tail_mean = np.mean(lower_tail)
            lower_tail_var = np.var(lower_tail)

        #tail_weight_ratio = np.sum((values > (median + 1.5 * iqr)) | (values < (median - 1.5 * iqr))) / count
        #tail_weight_ratio = np.sum((values >= p95) | (values <= p5)) / count
        if num_unique <= 2: # should this be 1 or 2?
            tail_weight_ratio = 0
        else:
            tail_weight_ratio = np.sum((values >= p95) | (values <= p5)) / count
        tail_length_ratio_95 = tail_length_95 / range_ if range_ > 0 else 0
        tail_length_ratio_05 = tail_length_05 / range_ if range_ > 0 else 0
        excess_kurtosis = kurt - 3

        # Outlier detection
        zscores = zscore(values)
        outliers_zscore = np.abs(zscores) > 3
        count_outliers_zscore = np.sum(outliers_zscore)
        count_outliers_zscore_prop = count_outliers_zscore / count

        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1   
        outliers_iqr = (values < (Q1 - 1.5 * IQR)) | (values > (Q3 + 1.5 * IQR))
        count_outliers_iqr = np.sum(outliers_iqr)
        count_outliers_iqr_prop = count_outliers_iqr / count

        outliers_tukey = (values > (Q3 + 3 * IQR)) | (values < (Q1 - 3 * IQR))
        count_outliers_tukey = np.sum(outliers_tukey)
        count_outliers_tukey_prop = count_outliers_tukey / count

        outliers_qq, long_tail_qq = classify_qq_outliers(values)
        count_positive_qq = (outliers_qq + long_tail_qq) / count

        # Goodness of fit parameters
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="scipy.stats.shapiro: For N > 5000")
            try:
                fitted_mean, fitted_std_dev = norm.fit(values)
                ks_stat_norm, ks_p_value_norm = (kstest(values, 'norm', args=(fitted_mean, fitted_std_dev)) 
                                                 if num_unique > 2 else (1, 0))
                shapiro_stat, shapiro_p_value = (shapiro(values) 
                                                 if num_unique > 2 else (0, 0))
            except Exception as e:
                ks_stat_norm, ks_p_value_norm = np.nan, np.nan
                shapiro_stat, shapiro_p_value = np.nan, np.nan
                print(f"Error during statistical tests: {e}")

        return {
            #'Count': count,
            'Unique_Count': num_unique,
            'Mean': mean,
            'Median': median,
            'Std_Dev': std_dev,
            'IQR': iqr,
            'Skewness': skewness,
            'Kurtosis': kurt,
            'Min': min_value,
            'Max': max_value,
            'Range': range_,
            'Upper_Tail': upper_tail_diff,
            'Lower_Tail': lower_tail_diff,
            'Extreme_Tail_95': tail_length_95,
            'Extreme_Tail_99': tail_length_99,
            'Extreme_Tail_05': tail_length_05,
            'Extreme_Tail_01': tail_length_01,
            'Upper_Tail_Mean': upper_tail_mean,
            'Upper_Tail_Var': upper_tail_var,
            'Lower_Tail_Mean': lower_tail_mean,
            'Lower_Tail_Var': lower_tail_var,
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

    # Apply feature computation to each column
    feature_vectors = df_scaled.apply(compute_features, axis=0).tolist()

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
        mean = np.mean(values)
        median = np.median(values)
        std_dev = np.std(values)
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        skewness = skew(values) if num_unique > 2 else 0
        kurt = kurtosis(values) if num_unique > 2 else 0
        min_value = np.min(values)
        max_value = np.max(values)
        range_ = max_value - min_value
        p95, p5, p99, p1 = np.percentile(values, [95, 5, 99, 1])
        upper_tail_diff = p99 - p95
        lower_tail_diff = p5 - p1
        tail_length_95 = max_value - p95
        tail_length_99 = max_value - p99
        tail_length_05 = p5 - min_value
        tail_length_01 = p1 - min_value
        upper_tail = values[values >= p95]
        lower_tail = values[values <= p5]

        if num_unique <= 2: 
            upper_tail_mean = 0
            upper_tail_var = 0
            lower_tail_mean = 0
            lower_tail_var = 0
        else: 
            upper_tail_mean = np.mean(upper_tail)
            upper_tail_var = np.var(upper_tail)
            lower_tail_mean = np.mean(lower_tail)
            lower_tail_var = np.var(lower_tail)

        #tail_weight_ratio = np.sum((values > (median + 1.5 * iqr)) | (values < (median - 1.5 * iqr))) / count
        #tail_weight_ratio = np.sum((values >= p95)| (values >= p5)) / count
        if num_unique <= 2: # should this be 1 or 2?
            tail_weight_ratio = 0
        else:
            tail_weight_ratio = np.sum((values >= p95) | (values <= p5)) / count
        tail_length_ratio_95 = tail_length_95 / range_ if range_ > 0 else 0
        tail_length_ratio_05 = tail_length_05 / range_ if range_ > 0 else 0
        excess_kurtosis = kurt - 3

        # Outlier detection
        zscores = zscore(values)
        outliers_zscore = np.abs(zscores) > 3
        count_outliers_zscore = np.sum(outliers_zscore)
        count_outliers_zscore_prop = count_outliers_zscore / count

        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1   
        outliers_iqr = (values < (Q1 - 1.5 * IQR)) | (values > (Q3 + 1.5 * IQR))
        count_outliers_iqr = np.sum(outliers_iqr)
        count_outliers_iqr_prop = count_outliers_iqr / count

        outliers_tukey = (values > (Q3 + 3 * IQR)) | (values < (Q1 - 3 * IQR))
        count_outliers_tukey = np.sum(outliers_tukey)
        count_outliers_tukey_prop = count_outliers_tukey / count

        outliers_qq, long_tail_qq = classify_qq_outliers(values)
        count_positive_qq = (outliers_qq + long_tail_qq) / count

        # Goodness of fit parameters
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="scipy.stats.shapiro: For N > 5000")
            try:
                fitted_mean, fitted_std_dev = norm.fit(values)
                ks_stat_norm, ks_p_value_norm = (kstest(values, 'norm', args=(fitted_mean, fitted_std_dev)) 
                                                 if num_unique > 2 else (1, 0))
                shapiro_stat, shapiro_p_value = (shapiro(values) 
                                                 if num_unique > 2 else (0, 0))
            except Exception as e:
                ks_stat_norm, ks_p_value_norm = np.nan, np.nan
                shapiro_stat, shapiro_p_value = np.nan, np.nan
                print(f"Error during statistical tests: {e}")
            

        feature_vector = {
                #'Count': count,
                'Unique_Count': num_unique,
                'Mean': mean,
                'Median': median,
                'Std_Dev': std_dev,
                'IQR': iqr,
                'Skewness': skewness,
                'Kurtosis': kurt,
                'Min': min_value,
                'Max': max_value,
                'Range': range_,
                'Upper_Tail': upper_tail_diff,
                'Lower_Tail': lower_tail_diff,
                'Extreme_Tail_95': tail_length_95,
                'Extreme_Tail_99': tail_length_99,
                'Extreme_Tail_05': tail_length_05,
                'Extreme_Tail_01': tail_length_01,
                'Upper_Tail_Mean': upper_tail_mean,
                'Upper_Tail_Var': upper_tail_var,
                'Lower_Tail_Mean': lower_tail_mean,
                'Lower_Tail_Var': lower_tail_var,
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

def classify_qq_outliers(data, outlier_threshold=3, long_tail_threshold=3):
    
    #data = pd.to_numeric(data , errors='coerce').dropna()
    data = pd.to_numeric(pd.Series(data), errors='coerce').dropna()

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


def exensio_get_features_old(df):
    feature_keys = ['Count', 'Unique_Count', 'Mean', 'Median', 'Std_Dev', 'IQR', 'Skewness',
    'Kurtosis', 'Min', 'Max', 'Range', 'Upper_Tail', 'Lower_Tail', 'Extreme_Tail_95',
    'Extreme_Tail_99', 'Extreme_Tail_05', 'Extreme_Tail_01', 'Upper_Tail_Mean',
    'Upper_Tail_Var', 'Lower_Tail_Mean', 'Lower_Tail_Var', 'Tail_Weight_Ratio',
    'Tail_Length_Ratio_95', 'Tail_Length_Ratio_05', 'Excess_Kurtosis', 'P99', 'P1',
    'Outliers_Zscore', 'Outliers_Zscore_prop', 'Outliers_IQR', 'Outliers_IQR_prop',
    'Outliers_Tukey', 'Outliers_Tukey_prop', 'QQ Count', 'KS_Stat_norm', 'KS_P_value_norm',
    'Shapiro_Stat', 'Shapiro_P_value']
    feature_vectors = []   
    # scaler = StandardScaler()
    # normalized_data = scaler.fit_transform(df)
    # df_scaled = pd.DataFrame(normalized_data) 

    scaler = MinMaxScaler()
    minmax_data = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(minmax_data)

    for column in df_scaled.columns:  #iterate through each test  
        values = df_scaled[column].dropna()
        if values.empty:
            empty_vector = {key: 0 for key in feature_keys}
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="scipy.stats.shapiro: For N > 5000")
            try:
                fitted_mean, fitted_std_dev = norm.fit(values)
                if num_unique <= 2:
                    ks_stat_norm, ks_p_value_norm = 1, 0
                    shapiro_stat, shapiro_p_value = 0, 0
                else:
                    ks_stat_norm, ks_p_value_norm = kstest(values, 'norm', args=(fitted_mean, fitted_std_dev))
                    shapiro_stat, shapiro_p_value = shapiro(values)
            except Exception as e:
                ks_stat_norm, ks_p_value_norm = np.nan, np.nan
                shapiro_stat, shapiro_p_value = np.nan, np.nan
                print(f"Error during statistical tests: {e}")

        
        # warnings.simplefilter("ignore", category=RuntimeWarning)
        # fitted_mean, fitted_std_dev = norm.fit(values)
        # warnings.filterwarnings("ignore", message="scipy.stats.shapiro: For N > 5000")

        # if num_unique <= 2: 
        #     ks_stat_norm, ks_p_value_norm = 1, 0
        #     shapiro_stat, shapiro_p_value = 0, 0
        # else:
        #     ks_stat_norm, ks_p_value_norm = kstest(values, 'norm', args=(fitted_mean, fitted_std_dev))
        #     shapiro_stat, shapiro_p_value = shapiro(values)

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
                'Lower_Tail_Var': lower_tail_var,
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
