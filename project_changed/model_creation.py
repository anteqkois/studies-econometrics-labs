import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

from tests import test_heteroscedasticity, test_normality, test_vif, test_chow, test_ramsey_reset, test_runs
from helwig_model import hellwig_method
from corrective_methods import logarithmic_transformation, structural_break_correction, ramsey_reset_correction


def build_initial_model(X_encoded, y_data, verbose=True):

    if verbose:
        print("\n" + "="*60)
        print("BUDOWA POCZĄTKOWEGO MODELU OPARTEGO NA METODZIE HELLWIGA")
        print("="*60)
    
    # Wywołanie metody Hellwiga
    hellwig_vars = hellwig_method(X_encoded, y_data, threshold=0.1, verbose = verbose)
    
    # Przygotowanie danych dla modelu Hellwiga
    X_hellwig = X_encoded[hellwig_vars].copy()
    
    # Upewnienie się, że wszystkie dane są numeryczne
    X_hellwig = X_hellwig.astype(float)
    y_data_clean = y_data.astype(float)
    
    # Sprawdzenie i usunięcie wartości NaN
    valid_mask = ~(X_hellwig.isnull().any(axis=1) | y_data_clean.isnull())
    X_hellwig = X_hellwig[valid_mask]
    y_data_clean = y_data_clean[valid_mask]
    
    # Dodanie stałej (intercept)
    X_hellwig_with_const = sm.add_constant(X_hellwig)
    
    # Budowa modelu
    hellwig_model = sm.OLS(y_data_clean, X_hellwig_with_const).fit()
    
    if verbose:
        print(f"Model zbudowany z {len(hellwig_vars)} zmiennymi wybranymi metodą Hellwiga:")
        print("Wybrane zmienne:", hellwig_vars)
        
        print("\n" + "="*60)
        print("PODSUMOWANIE MODELU HELLWIGA")
        print("="*60)
        print(hellwig_model.summary())
    
    return hellwig_model, X_hellwig, y_data_clean, hellwig_vars


def run_diagnostic_tests(model, X_data, verbose=True):

    residuals = model.resid
    
    # 1. Podstawowe statystyki reszt
    if verbose:
        print(f"\n{'='*40}")
        print("PODSTAWOWE STATYSTYKI RESZT")
        print(f"{'='*40}")
        print(f"Liczba obserwacji: {len(residuals)}")
        print(f"Średnia reszt: {residuals.mean():.6f}")
        print(f"Mediana reszt: {residuals.median():.6f}")
        print(f"Odchylenie standardowe reszt: {residuals.std():.4f}")
        print(f"Minimum: {residuals.min():.4f}")
        print(f"Maksimum: {residuals.max():.4f}")
        print(f"Rozstęp: {residuals.max() - residuals.min():.4f}")
    
    # 2. Miary asymetrii i koncentracji
    skewness = residuals.skew()
    kurtosis = residuals.kurtosis()
    

    # 3. Testy normalności
    if verbose:
        print(f"\n{'='*40}")
        print("TESTY NORMALNOŚCI")
        print(f"{'='*40}")
    
    normality_results = test_normality(residuals, verbose=verbose)
    
    # 4. Wizualizacje normalności
    if verbose:
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Histogram z krzywą normalną
        ax1.hist(residuals, bins=30, density=True, alpha=0.7, color='lightblue', edgecolor='black')
        x_norm = np.linspace(residuals.min(), residuals.max(), 100)
        y_norm = stats.norm.pdf(x_norm, residuals.mean(), residuals.std())
        ax1.plot(x_norm, y_norm, 'r-', linewidth=2, label='Rozkład normalny')
        ax1.set_xlabel('Reszty')
        ax1.set_ylabel('Gęstość')
        ax1.set_title('Histogram reszt z krzywą normalną')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normalność)')
        ax2.grid(True, alpha=0.3)
        
        # P-P plot
        sorted_residuals = np.sort(residuals)
        theoretical_quantiles = stats.norm.cdf(sorted_residuals, residuals.mean(), residuals.std())
        empirical_quantiles = np.arange(1, len(sorted_residuals) + 1) / len(sorted_residuals)
        ax3.plot(theoretical_quantiles, empirical_quantiles, 'bo', alpha=0.6)
        ax3.plot([0, 1], [0, 1], 'r-', linewidth=2)
        ax3.set_xlabel('Teoretyczne percentyle')
        ax3.set_ylabel('Empiryczne percentyle')
        ax3.set_title('P-P Plot (Normalność)')
        ax3.grid(True, alpha=0.3)
        
        # Boxplot
        ax4.boxplot(residuals, vert=True)
        ax4.set_ylabel('Reszty')
        ax4.set_title('Boxplot reszt')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # 5. Test heteroskedastyczności
    if verbose:
        print(f"\n{'='*40}")
        print("TESTY HETEROSKEDASTYCZNOŚCI")
        print(f"{'='*40}")
    
    X_with_const = sm.add_constant(X_data) if 'const' not in X_data.columns else X_data
    heteroskedasticity_results = test_heteroscedasticity(residuals, X_with_const, verbose=verbose)
    

    return {
        'skewness': skewness,
        'kurtosis': kurtosis,
        'residuals': residuals,
        'normality_results': normality_results,
        'heteroskedasticity_results': heteroskedasticity_results,
    }


def test_model_stability(model, y_data, X_data, verbose=True):


    # Testowanie VIF dla zmiennych objaśniających (bez stałej)
    X_no_const = X_data.drop('const', axis=1) if 'const' in X_data.columns else X_data
    vif_results = test_vif(X_no_const, threshold=10.0, verbose=verbose)
    
    
    # Dla testu Chowa potrzeba określić punkt podziału
    n_obs = len(y_data)
    break_point = n_obs // 2
    

    # Przeprowadzenie testu Chowa
    X_with_const = sm.add_constant(X_no_const) if 'const' not in X_data.columns else X_data
    chow_results = test_chow(y_data, X_with_const, break_point, alpha=0.05, verbose=verbose)
    

    reset_results = test_ramsey_reset(model, alpha=0.05, power=3, verbose=verbose)
    
    # TEST LICZBY SERII - LOSOWOŚĆ RESZT
    
    runs_results = test_runs(model.resid, alpha=0.05, verbose=verbose)
    
    return {
        'break_point': break_point,
        'vif_results': vif_results,
        'chow_results': chow_results,
        'reset_results': reset_results,
        'runs_results': runs_results,
    }

def print_test_summary(test_results, test_type="Testy diagnostyczne"):
    """
    Wyświetla krótkie podsumowanie wyników testów
    """
    print(f"\n{'='*50}")
    print(f"PODSUMOWANIE: {test_type.upper()}")
    print(f"{'='*50}")
    
    if 'summary' in test_results:
        for result in test_results['summary']:
            status_icon = "✓" if result['interpretation'].startswith("OK") else "⚠" if result['interpretation'].startswith("UWAGA") else "✗"
            print(f"{status_icon} {result['test']}: {result['interpretation']}")
    else:
        print("Brak dostępnego podsumowania")
    print(f"{'='*50}")




def build_and_test_models(X_encoded, y_data, categorical_cols, NUM_COLS, BIN_COLS, df_clean):
    print("ROZPOCZĘCIE BUDOWY I TESTOWANIA MODELI")
    print("="*60)
    

    current_model, X_hellwig, y_data_clean, hellwig_vars = build_initial_model(X_encoded, y_data, False)
    

    diagnostic_results = run_diagnostic_tests(current_model, X_hellwig, False)
    stability_results = test_model_stability(current_model, y_data_clean, X_hellwig, False)
    

    # METODA NAPRAWCZA 1: Transformacja logarytmiczna    
    current_model, y_log = logarithmic_transformation(X_hellwig, y_data_clean, False)
    
    # Testy po transformacji logarytmicznej
    diagnostic_results = run_diagnostic_tests(current_model, X_hellwig, False)
    stability_results = test_model_stability(current_model, y_log, X_hellwig, False)
    
    # METODA NAPRAWCZA 2: Przełamanie strukturalne    
    current_model = structural_break_correction(y_log, X_hellwig, stability_results['break_point'], False)
    
    # Testy po przełamaniu strukturalnym
    X_interactions = X_hellwig.copy()
    group_dummy = (X_interactions.index >= stability_results['break_point']).astype(int)
    for col in X_hellwig.columns:
        X_interactions[f'{col}_group2'] = X_hellwig[col] * group_dummy
    X_interactions['group_2'] = group_dummy
    
    diagnostic_results = run_diagnostic_tests(current_model, X_interactions, False)
    stability_results = test_model_stability(current_model, y_log, X_interactions, False)
    
    # METODA NAPRAWCZA 3: Korekta Ramsey RESET
    current_model = ramsey_reset_correction(X_hellwig, y_log)
    
    # Finalne testy
    X_advanced = X_hellwig.copy()
    continuous_vars = ['CPU_freq', 'SecondaryStorage']
    for var in continuous_vars:
        if var in X_hellwig.columns:
            var_std = (X_hellwig[var] - X_hellwig[var].mean()) / X_hellwig[var].std()
            X_advanced[f'{var}_squared'] = var_std ** 2
    
    if 'SecondaryStorage' in X_hellwig.columns:
        X_advanced['Storage_log'] = np.log(X_hellwig['SecondaryStorage'] + 1)
    
    interactions = [
        ('CPU_freq', 'Touchscreen'),
        ('SecondaryStorage', 'IPSpanel')
    ]
    
    for cont_var, bin_var in interactions:
        if cont_var in X_hellwig.columns and bin_var in X_hellwig.columns:
            var_std = (X_hellwig[cont_var] - X_hellwig[cont_var].mean()) / X_hellwig[cont_var].std()
            X_advanced[f'{cont_var}_x_{bin_var}'] = var_std * X_hellwig[bin_var]
    
    diagnostic_results = run_diagnostic_tests(current_model, X_advanced)
    stability_results = test_model_stability(current_model, y_log, X_advanced)
    
    # print(f"\n{'='*60}")
    # print("ZAKOŃCZENIE BUDOWY I TESTOWANIA MODELI")
    # print(f"{'='*60}")
    # print(f"Końcowy model R²: {current_model.rsquared:.4f}")
    # print(f"Końcowy model Adjusted R²: {current_model.rsquared_adj:.4f}")
    # print(f"Końcowy model AIC: {current_model.aic:.4f}")
    # print(f"Końcowy model BIC: {current_model.bic:.4f}")
    
    return current_model