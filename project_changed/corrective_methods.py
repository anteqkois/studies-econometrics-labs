import numpy as np
import statsmodels.api as sm


def logarithmic_transformation(X_hellwig, y_data_clean, verbose=True):
    
    if verbose:
        print(f"\n{'='*60}")
        print("ZASTOSOWANIE METODY NAPRAWCZEJ 1: TRANSFORMACJA LOGARYTMICZNA")
        print(f"{'='*60}")

    y_log = np.log(y_data_clean)
    X_hellwig_with_const = sm.add_constant(X_hellwig)
    hellwig_model_log = sm.OLS(y_log, X_hellwig_with_const).fit()
    

    return hellwig_model_log, y_log


def structural_break_correction(y_log, X_hellwig, break_point, verbose=True):

    if verbose:
        print(f"\n{'='*60}")
        print("ZASTOSOWANIE METODY NAPRAWCZEJ 2: PRZEŁAMANIE STRUKTURALNE")
        print(f"{'='*60}")

    # Tworzenie interakcji wszystkich zmiennych z dummy grupy
    X_interactions = X_hellwig.copy()
    group_dummy = (X_interactions.index >= break_point).astype(int)
    
    # Dodanie interakcji
    for col in X_hellwig.columns:
        X_interactions[f'{col}_group2'] = X_hellwig[col] * group_dummy
    
    X_interactions['group_2'] = group_dummy
    X_inter_with_const = sm.add_constant(X_interactions)
    
    # Model z interakcjami
    interaction_model = sm.OLS(y_log, X_inter_with_const).fit()
    
    return interaction_model


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