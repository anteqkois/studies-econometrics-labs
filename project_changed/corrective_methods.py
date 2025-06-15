import numpy as np
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
    # X_interactions['group_2'] = group_dummy

    return build_model_fn(X_interactions, y_log, verbose=True)

def ramsey_reset_correction(X_hellwig, y_log, verbose=True):
    if verbose:
        print(f"\n{'='*50}")
        print("METODA NAPRAWCZA: RAMSEY RESET")
        print(f"{'='*50}")
    
    X_advanced = X_hellwig.copy()
    
    # 1. Kwadraty najważniejszych zmiennych ciągłych
    continuous_vars = ['CPU_freq', 'SecondaryStorage']
    for var in continuous_vars:
        if var in X_hellwig.columns:
            # Standaryzacja przed potęgowaniem
            var_std = (X_hellwig[var] - X_hellwig[var].mean()) / X_hellwig[var].std()
            X_advanced[f'{var}_squared'] = var_std ** 2
    
    # 2. Logarytmy dla zmiennych o rozkładzie skośnym
    if 'SecondaryStorage' in X_hellwig.columns:
        X_advanced['Storage_log'] = np.log(X_hellwig['SecondaryStorage'] + 1)
    
    # 3. Proste interakcje między zmiennymi ciągłymi a binarnymi
    interactions = [
        ('CPU_freq', 'Touchscreen'),
        ('SecondaryStorage', 'IPSpanel')
    ]
    
    for cont_var, bin_var in interactions:
        if cont_var in X_hellwig.columns and bin_var in X_hellwig.columns:
            # Standaryzacja zmiennej ciągłej przed interakcją
            var_std = (X_hellwig[cont_var] - X_hellwig[cont_var].mean()) / X_hellwig[cont_var].std()
            X_advanced[f'{cont_var}_x_{bin_var}'] = var_std * X_hellwig[bin_var]
    
    # Budowa ulepszonego modelu
    X_advanced_with_const = sm.add_constant(X_advanced)
    advanced_model = sm.OLS(y_log, X_advanced_with_const).fit()
    
    return advanced_model