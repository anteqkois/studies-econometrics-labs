import pandas as pd
from model_creation import build_and_test_models
from data_preparation import preprocess_data

# Wczytanie danych
df_raw = pd.read_csv("kois_mazur_projekt_train_data.csv")

# Przetwarzanie danych treningowych
X_encoded, y_data, top_k_dict, fitted_top_categories, df_clean = preprocess_data(
    df_raw,
    is_train=True
)

# Grupy dummy (prefiksowane nazwy)
categorical_cols = ['Company', 'OS', 'Screen', 'CPU_company', 'PrimaryStorageType', 'GPU_company']
DUMMY_GROUPS = {
    cat: [col for col in X_encoded.columns if col.startswith(cat + "_")]
    for cat in categorical_cols
}

# Przekazanie do pipeline'u modelowego
build_and_test_models(
    X_encoded, y_data,
    categorical_cols=categorical_cols,
    NUM_COLS=[
        'Inches', 'Ram', 'Weight',
        'ScreenW', 'CPU_freq',
        'PrimaryStorage', 'SecondaryStorage'
    ],
    BIN_COLS=['Touchscreen', 'IPSpanel', 'RetinaDisplay'],
    DUMMY_GROUPS=DUMMY_GROUPS,
    df_clean=df_clean
)
