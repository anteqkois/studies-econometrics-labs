import pandas as pd

TOP_K = {
    'Company': 6,
    'OS': 5
}

DROP_COLS = ['Product', 'TypeName', 'CPU_model', 'ScreenH', 'GPU_model', 'SecondaryStorageType']
BIN_COLS = ['Touchscreen', 'IPSpanel', 'RetinaDisplay']
BINARY_MAP = {'Yes': 1, 'No': 0}

CAT_TOPK_COLS = ['Company', 'OS']
CAT_SMALL_COLS = ['Screen', 'CPU_company', 'PrimaryStorageType', 'GPU_company']
NUM_COLS = ['Inches', 'Ram', 'Weight', 'ScreenW', 'CPU_freq', 'PrimaryStorage', 'SecondaryStorage']

def iqr_bounds(series, k=1.5):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

def iqr_filter(df: pd.DataFrame, cols, k=1.5):
    mask = pd.Series(True, index=df.index)
    for c in cols:
        low, high = iqr_bounds(df[c], k)
        mask &= df[c].between(low, high)
    return df[mask].copy()

def create_dummy_vars(df, categorical_cols):
    df_dummies = df.copy()
    for col in categorical_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df_dummies = pd.concat([df_dummies, dummies], axis=1)
        df_dummies = df_dummies.drop(columns=[col])
    return df_dummies

def preprocess_data(df, is_train=True, top_k_dict=None, fitted_top_categories=None):
    df = df.copy()
    df.drop(columns=DROP_COLS, errors='ignore', inplace=True)
    df[BIN_COLS] = df[BIN_COLS].replace(BINARY_MAP).astype(int)

    if is_train:
        top_k_dict = {}
        fitted_top_categories = {}
        for col in CAT_TOPK_COLS:
            vc = df[col].value_counts()
            top_k = vc.nlargest(TOP_K[col])
            fitted_top_categories[col] = list(top_k.index)
            df[col] = df[col].where(df[col].isin(top_k.index), 'Other')
            top_k_dict[col] = TOP_K[col]
    else:
        for col in CAT_TOPK_COLS:
            df[col] = df[col].where(df[col].isin(fitted_top_categories[col]), 'Other')

    # Outlier filtering
    df_clean = iqr_filter(df, ['Weight', 'Inches'])

    y = df_clean['Price_euros'] if 'Price_euros' in df_clean.columns else None
    categorical_cols = CAT_TOPK_COLS + CAT_SMALL_COLS
    X_data = df_clean[NUM_COLS + BIN_COLS + categorical_cols].copy()
    X_encoded = create_dummy_vars(X_data, categorical_cols)

    return X_encoded, y, top_k_dict, fitted_top_categories, df_clean
