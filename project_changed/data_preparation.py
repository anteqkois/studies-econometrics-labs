import pandas as pd

# Funkcja do kodowania zmiennych kategorycznych jako dummy variables
def create_dummy_vars(df, categorical_cols):
    df_dummies = df.copy()
    
    for col in categorical_cols:
        # Tworzymy zmienne dummy dla każdej kategorii (pomijamy pierwszą jako referencyjną)
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df_dummies = pd.concat([df_dummies, dummies], axis=1)
        df_dummies = df_dummies.drop(columns=[col])
    
    return df_dummies

TOP_K = {
    'Company': 6,
    'OS': 5
}

def preprocess_data(df, is_train=True, top_k_dict=None, fitted_top_categories=None):
    # Kopia danych
    df = df.copy()

    # Kolumny do usunięcia
    DROP_COLS = ['Product', 'TypeName', 'CPU_model', 'ScreenH', 'GPU_model', 'SecondaryStorageType']
    df.drop(columns=DROP_COLS, errors='ignore', inplace=True)

    # Binarne
    BIN_COLS = ['Touchscreen', 'IPSpanel', 'RetinaDisplay']
    BINARY_MAP = {'Yes': 1, 'No': 0}
    df[BIN_COLS] = df[BIN_COLS].replace(BINARY_MAP).astype(int)

    # Kolumny kategorii do przetworzenia
    CAT_TOPK_COLS = ['Company', 'OS']
    CAT_SMALL_COLS = ['Screen', 'CPU_company', 'PrimaryStorageType', 'GPU_company']
    categorical_cols = CAT_TOPK_COLS + CAT_SMALL_COLS

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

    # Zmienna celu
    y = df['Price_euros'] if 'Price_euros' in df.columns else None

    # Kolumny numeryczne
    NUM_COLS = [
        'Inches', 'Ram', 'Weight',
        'ScreenW', 'CPU_freq',
        'PrimaryStorage', 'SecondaryStorage'
    ]

    # Dane X
    X_data = df[NUM_COLS + BIN_COLS + categorical_cols].copy()

    # Kodowanie dummy
    X_encoded = create_dummy_vars(X_data, categorical_cols)

    return X_encoded, y, top_k_dict, fitted_top_categories
