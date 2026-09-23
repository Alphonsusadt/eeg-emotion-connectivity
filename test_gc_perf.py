import time
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

INPUT_DIR  = r"D:\Skripsi\new_data\00_preprocessing\output\preprocessed_eeg"
sample_path = f"{INPUT_DIR}/subject_15/session_20130709/trial_01.npy"
eeg = np.load(sample_path)

def check_stationarity(ts):
    from statsmodels.tsa.stattools import adfuller
    try:
        result = adfuller(ts, autolag='AIC')
        return result[1] < 0.05
    except:
        return False

def make_stationary(data):
    n_tp, n_ch = data.shape
    result = np.zeros_like(data)
    for ch in range(n_ch):
        ts = data[:, ch]
        if not check_stationarity(ts):
            ts = np.diff(ts)
            result[:len(ts), ch] = ts
            result[len(ts):, ch] = ts[-1]
        else:
            result[:, ch] = ts
    return result

def normalize_channels(data):
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std[std < 1e-10] = 1.0
    return (data - mean) / std

def preprocess_for_gc(eeg_data):
    data = eeg_data.T
    data = normalize_channels(data)
    data = make_stationary(data)
    return data

def select_optimal_lag(data_pair, lag_min=2, lag_max=5):
    best_aic = np.inf
    best_lag = lag_min
    for lag in range(lag_min, lag_max + 1):
        try:
            model = VAR(data_pair)
            result = model.fit(lag)
            if result.aic < best_aic:
                best_aic = result.aic
                best_lag = lag
        except:
            continue
    return best_lag

def compute_gc_pairwise(data, lag_min=2, lag_max=5):
    n_ch = data.shape[1]
    gc_matrix = np.zeros((n_ch, n_ch))
    p_matrix = np.ones((n_ch, n_ch))
    lag_matrix = np.zeros((n_ch, n_ch), dtype=int)
    for i in range(n_ch):
        for j in range(n_ch):
            if i == j:
                continue
            try:
                test_data = data[:, [j, i]]
                opt_lag = select_optimal_lag(test_data, lag_min, lag_max)
                lag_matrix[i, j] = opt_lag
                result = grangercausalitytests(test_data, maxlag=opt_lag, verbose=False)
                f_stat = result[opt_lag][0]['ssr_ftest'][0]
                p_val = result[opt_lag][0]['ssr_ftest'][1]
                gc_matrix[i, j] = f_stat
                p_matrix[i, j] = p_val
            except Exception as e:
                continue
    return gc_matrix, p_matrix, lag_matrix

t0 = time.time()
data = preprocess_for_gc(eeg)
gc_raw, p_mat, lag_mat = compute_gc_pairwise(data)
t1 = time.time()
print(f"Trial processed in {t1-t0:.1f}s")
