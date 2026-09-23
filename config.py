"""
Konfigurasi Global Pipeline GC + PDC
======================================
Pipeline lengkap: Raw EEG -> Preprocessing -> GC + PDC -> Phase 1-6
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════
DATASET_PATH = r"D:\Skripsi\dataset\SEED\SEED\SEED_EEG"
CHANNEL_FILE = r"D:\Skripsi\dataset\SEED\SEED\channel_62_pos.locs"
RAW_EEG_PATH = os.path.join(DATASET_PATH, "Preprocessed_EEG")
EXTRACTED_FEATURES_PATH = os.path.join(DATASET_PATH, "ExtractedFeatures_1s")

# ═══════════════════════════════════════════════════════════
# OUTPUT DIRECTORIES
# ═══════════════════════════════════════════════════════════
PREPROCESS_DIR = os.path.join(BASE_DIR, "00_preprocessing", "output", "preprocessed_eeg")
GC_OUTPUT_DIR = os.path.join(BASE_DIR, "01_granger_causality", "output", "gc_matrices")
GC_FIGURES_DIR = os.path.join(BASE_DIR, "01_granger_causality", "output", "figures")
PDC_OUTPUT_DIR = os.path.join(BASE_DIR, "02_pdc", "output", "pdc_matrices")
PDC_FIGURES_DIR = os.path.join(BASE_DIR, "02_pdc", "output", "figures")
PHASE1_DIR = os.path.join(BASE_DIR, "phase_1_data_structuring")
PHASE2_DIR = os.path.join(BASE_DIR, "phase_2_statistics")
PHASE3_DIR = os.path.join(BASE_DIR, "phase_3_feature_engineering")
PHASE4_DIR = os.path.join(BASE_DIR, "phase_4_classification")
PHASE5_DIR = os.path.join(BASE_DIR, "phase_5_loso")
PHASE6_DIR = os.path.join(BASE_DIR, "phase_6_deep_learning")

# ═══════════════════════════════════════════════════════════
# STEP 0: PREPROCESSING
# ═══════════════════════════════════════════════════════════
ORIGINAL_FS = 200              # Hz (sampling rate asli SEED)
TARGET_FS = 100                # Hz (downsample target, factor 2)
BANDPASS_LOW = 1.0             # Hz
BANDPASS_HIGH = 45.0           # Hz
BANDPASS_ORDER = 4             # Filter order
NOTCH_FREQ = 50.0              # Hz (line noise)
NOTCH_Q = 30.0                 # Notch quality factor

# ═══════════════════════════════════════════════════════════
# STEP 1: GRANGER CAUSALITY (62 channels, concatenate trials)
# ═══════════════════════════════════════════════════════════
GC_N_CHANNELS = 62                 # Full 62 channels for GC
GC_MAX_LAG = 10
GC_ALPHA = 0.05
GC_THRESHOLD_METHOD = "fdr"        # "fdr", "bonferroni", "percentile"
GC_PERCENTILE = 90                 # Top 10% connections only
GC_GRANULARITY = "concatenated"     # "concatenated" = per subject-session-emotion (135 samples)

# ═══════════════════════════════════════════════════════════
# STEP 2: PDC (ROI channels, per-trial)
# ═══════════════════════════════════════════════════════════
PDC_MAX_ORDER = 15                 # AIC selects optimal from this range
PDC_MIN_ORDER = 3
PDC_FREQUENCY_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}
PDC_GRANULARITY = "per_trial"      # "per_trial" = per trial (~675 samples)

# ROI channels for PDC (emotion-relevant regions)
# Prefrontal + Frontal + Temporal + Central + Parietal key channels
ROI_CHANNELS = [
    # Prefrontal (emotion regulation)
    "Fp1", "Fp2", "AF3", "AF4",
    # Frontal (executive function, top-down regulation)
    "F3", "F4", "Fz", "F7", "F8", "FC1", "FC2", "FCz",
    # Frontal-temporal (emotional processing)
    "FT7", "FT8", "FC5", "FC6", "FC3", "FC4",
    # Temporal (limbic connectivity)
    "T7", "T8", "TP7", "TP8",
    # Central (sensorimotor integration)
    "C3", "C4", "Cz", "C1", "C2",
    # Parietal (attention to emotion)
    "P3", "P4", "Pz", "CP1", "CP2",
]
PDC_N_CHANNELS = len(ROI_CHANNELS)  # 30 channels

# ═══════════════════════════════════════════════════════════
# GENERAL
# ═══════════════════════════════════════════════════════════
N_CHANNELS = 62
N_SUBJECTS = 15
EMOTION_MAP = {-1: "negative", 0: "neutral", 1: "positive"}
LABEL_MAP = {-1: 0, 0: 1, 1: 2}       # For sklearn: 0=Neg, 1=Neu, 2=Pos
EMOTION_NAMES = ["Negative", "Neutral", "Positive"]

# Subject-session mapping (SEED dataset)
SUBJECT_SESSIONS = {
    1: ["20131027", "20131030", "20131107"],
    2: ["20140404", "20140413", "20140419"],
    3: ["20140603", "20140611", "20140629"],
    4: ["20140621", "20140702", "20140705"],
    5: ["20140411", "20140418", "20140506"],
    6: ["20130712", "20131016", "20131113"],
    7: ["20131027", "20131030", "20131106"],
    8: ["20140511", "20140514", "20140521"],
    9: ["20140620", "20140627", "20140704"],
    10: ["20131130", "20131204", "20131211"],
    11: ["20140618", "20140625", "20140630"],
    12: ["20131127", "20131201", "20131207"],
    13: ["20140527", "20140603", "20140610"],
    14: ["20140601", "20140615", "20140627"],
    15: ["20130709", "20131016", "20131105"],
}

# Trial labels (same for all subjects/sessions)
# -1=negative, 0=neutral, 1=positive
TRIAL_LABELS = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

# Channel names (62 channels)
CHANNEL_NAMES = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz", "C2", "C4",
    "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1", "O1", "Oz",
    "O2", "CB2",
]
