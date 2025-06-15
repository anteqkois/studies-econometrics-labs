import numpy as np
from scipy.stats import skew
import statsmodels.api as sm
from tests import test_t_student_significance
from build_model import build_model

def logarithmic_transformation(X_data, y_data_clean, build_model_fn, verbose=True):
    
    if verbose:
        print(f"\n{'='*60}")
        print("ZASTOSOWANIE METODY NAPRAWCZEJ: TRANSFORMACJA LOGARYTMICZNA")
        print(f"{'='*60}")

    y_log = np.log(y_data_clean)
    
    return build_model_fn(X_data, y_log, verbose=True)

def run_test_t_student_significance_and_remove(model, X_data, y_data_clean, verbose=True):
    if verbose:
        print(f"\n{'='*60}")
        print("ZASTOSOWANIE METODY NAPRAWCZEJ: USUNIĘCIE NIEISTOTYCH ZMIENNYCH")
        print(f"{'='*60}")

    significance_results = test_t_student_significance(model, verbose=verbose)
    
    if verbose:
        print(f"Usuwanie kolumn na podstawie testu t-Studenta: {significance_results['insignificant']}")
        print(f"Liczba kolumn przed usunięciem: {X_data.shape[1]}")
    
    X_data = X_data.drop(columns=significance_results['insignificant'])

    if verbose:
        print(f"Liczba kolumn po usunięciu: {X_data.shape[1]}")

    model, X_data_with_const, y_data_clean = build_model(X_data, y_data_clean, verbose=verbose)
    
    return model, X_data

def structural_break_correction(y_log, X_data, break_point, build_model_fn, verbose=True):
    if verbose:
        print(f"\n{'='*60}")
        print("ZASTOSOWANIE METODY NAPRAWCZEJ 2: PRZEŁAMANIE STRUCTURALNE")
        print(f"{'='*60}")

    # Tworzenie dummy wskazującego drugą grupę
    group_dummy = (X_data.index >= break_point).astype(int)

    # Kopia danych + interakcje tylko dla sensownych kolumn
    X_interactions = X_data.copy()

    for col in X_data.columns:
        # Pomiń kolumnę 'const' i ewentualnie wcześniej dodaną 'group_2'
        if col.lower() == 'const' or col.lower() == 'group_2':
            continue
        X_interactions[f'{col}_group2'] = X_data[col] * group_dummy

    # Dodaj tylko raz group_2 (bez interakcji)
    X_interactions['group_2'] = group_dummy

    return build_model_fn(X_interactions, y_log, verbose=True)

def ramsey_reset_correction(X_data, y_log, build_model_fn, verbose=True):
    if verbose:
        print(f"\n{'='*50}")
        print("METODA NAPRAWCZA: RAMSEY RESET")
        print(f"{'='*50}")
    
    X_advanced = X_data.copy()
    
    # 1. Kwadraty najważniejszych zmiennych ciągłych
    continuous_vars = ['CPU_freq', 'ScreenW']
    for var in continuous_vars:
        if var in X_data.columns:
            # Standaryzacja przed potęgowaniem
            var_std = (X_data[var] - X_data[var].mean()) / X_data[var].std()
            X_advanced[f'{var}_squared'] = var_std ** 2
    
    x_columns_to_log = []
    for col in X_data.select_dtypes(include='number').columns:
        unique_vals = X_data[col].dropna().unique()
        if set(unique_vals).issubset({0, 1}):
            continue  # pomiń dummy 0/1
        s = skew(X_data[col].dropna())
        if abs(s) > 1:
            x_columns_to_log.append(col)
            print(f"{col} ma silną skośność: {s:.2f}")
    
    # 2. Logarytmy dla zmiennych o rozkładzie skośnym i zamieniamy orginały
    for col in x_columns_to_log:
        X_advanced[f'{col}_log'] = np.log(X_data[col] + 1)
        if col in X_advanced.columns:
            X_advanced.drop(columns=col, inplace=True)
    
    # 3. Proste interakcje między zmiennymi ciągłymi a binarnymi (jeśli zm. kategoryczne, to tylko jedna zm. z danej kategorii)
    interactions = [
        # DOBRE 1
        # ('CPU_freq', 'PrimaryStorageType_SSD'),
        # ('ScreenW', 'GPU_company_Intel'),
        # DOBRE 2
        ('CPU_freq', 'Company_Lenovo'),
        ('ScreenW', 'GPU_company_Intel'),
    ]
    
    for cont_var, bin_var in interactions:
        if cont_var in X_data.columns and bin_var in X_data.columns:
            # Standaryzacja zmiennej ciągłej przed interakcją
            var_std = (X_data[cont_var] - X_data[cont_var].mean()) / X_data[cont_var].std()
            X_advanced[f'{cont_var}_x_{bin_var}'] = var_std * X_data[bin_var]
    
    # Budowa ulepszonego modelu
    return build_model_fn(X_advanced, y_log, verbose=verbose)

# def ramsey_reset_correction(
#     X_data,
#     y_data,
#     build_model_fn,
#     # continuous_vars=None,
#     # interactions=None,
#     verbose=True
# ):
#     if verbose:
#         print(f"\n{'='*50}")
#         print("METODA NAPRAWCZA: RAMSEY RESET")
#         print(f"{'='*50}")
        
#     # print(auto_detect_vars(X_data))
#     continuous_vars, interactions = auto_detect_vars(X_data, y_data, 2, 0, 0)
    
#     X_advanced = X_data.copy()

#     # Domyślne zmienne, jeśli nie podano
#     if continuous_vars is None:
#         continuous_vars = [col for col in X_data.columns if pd.api.types.is_numeric_dtype(X_data[col])]

#     if interactions is None:
#         interactions = []

#     # 1. Kwadraty zmiennych ciągłych
#     for var in continuous_vars:
#         if var in X_data.columns:
#             var_std = (X_data[var] - X_data[var].mean()) / X_data[var].std()
#             X_advanced[f'{var}_squared'] = var_std ** 2

#     # 2. Logarytmy tylko dla tych zmiennych, które są dodatnie
#     for var in continuous_vars:
#         if var in X_data.columns and (X_data[var] >= 0).all():
#             X_advanced[f'{var}_log'] = np.log(X_data[var] + 1)

#     # 3. Interakcje między ciągłymi a binarnymi
#     for cont_var, bin_var in interactions:
#         if cont_var in X_data.columns and bin_var in X_data.columns:
#             var_std = (X_data[cont_var] - X_data[cont_var].mean()) / X_data[cont_var].std()
#             X_advanced[f'{cont_var}_x_{bin_var}'] = var_std * X_data[bin_var]

#     # Budowa nowego modelu
#     return build_model_fn(X_advanced, y_data, verbose=True)