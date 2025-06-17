import pandas as pd
from model_creation import build_and_test_models
from data_preparation import preprocess_data, compare_columns

# Wczytanie danych
df_raw_train = pd.read_csv("kois_mazur_projekt_train_data.csv")
df_raw_test = pd.read_csv("kois_mazur_projekt_test_data.csv")

# Przetwarzanie danych treningowych
X_encoded_train, y_data_train, top_k_dict, fitted_top_categories, df_clean = preprocess_data(
    df_raw_train,
    is_train=True
)

# Grupy dummy (prefiksowane nazwy)
categorical_cols = ['Company', 'OS', 'Screen', 'CPU_company', 'PrimaryStorageType', 'GPU_company']
DUMMY_GROUPS = {
    cat: [col for col in X_encoded_train.columns if col.startswith(cat + "_")]
    for cat in categorical_cols
}

X_encoded_test, y_data_test, top_k_dict_test, fitted_top_categories_test, df_clean_test = preprocess_data(df_raw_test, False, top_k_dict, fitted_top_categories)

compare_columns(X_encoded_train, X_encoded_test, label1="train", label2="test")

# Przekazanie do pipeline'u modelowego
build_and_test_models(X_encoded_train, y_data_train, DUMMY_GROUPS, X_encoded_test, y_data_test)
