import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

from tests import test_heteroscedasticity, test_normality, test_vif, test_chow, test_ramsey_reset, test_runs, test_t_student_significance
from helwig_model import hellwig_method
from corrective_methods import logarithmic_transformation, run_test_t_student_significance_and_remove_v2, structural_break_correction, add_structural_break_columns, ramsey_reset_correction, run_test_t_student_significance_and_remove
from build_model import build_model, build_weighted_model
from charts import plot_correlation_heatmap
from predictions import calculate_forecast_errors
from data_preparation import compare_columns

def build_initial_model(X_encoded, y_data, DUMMY_GROUPS, verbose=True):
    if verbose:
        print("\n" + "="*60)
        print("BUDOWA POCZĄTKOWEGO MODELU OPARTEGO NA METODZIE HELLWIGA")
        print("="*60 + "\n")
    
    # Wywołanie metody Hellwiga (pomija zmienne kategoryczne)
    hellwig_vars = hellwig_method(X_encoded, y_data, DUMMY_GROUPS, threshold=0.1, verbose=verbose)

    # Dodanie wszystkich zmiennych kategorycznych
    all_dummy_vars = set()
    for group_vars in DUMMY_GROUPS.values():
        all_dummy_vars.update(group_vars)

    all_vars = list(set(hellwig_vars).union(all_dummy_vars))
    all_vars = [var for var in all_vars if var in X_encoded.columns]  # zabezpieczenie
    
    # Przygotowanie danych do modelu
    X_all_vars = X_encoded[all_vars].copy()
    X_all_vars = X_all_vars.astype(float)
    y_data_clean = y_data.astype(float)

    # Usunięcie wierszy z NaN
    valid_mask = ~(X_all_vars.isnull().any(axis=1) | y_data_clean.isnull())
    X_all_vars = X_all_vars[valid_mask]
    y_data_clean = y_data_clean[valid_mask]

    # Budowa modelu
    model, X_data_with_const, y_data_clean = build_model(X_all_vars, y_data_clean, verbose=verbose)

    return model, X_all_vars, y_data_clean, hellwig_vars, all_dummy_vars, all_vars, 


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

def run_test_t_student_significance(model, verbose=True):
    significance_results = test_t_student_significance(model, verbose=verbose)

def align_test_columns(X_train, X_test):
    """
    Dopasowuje testowe dane do struktury danych treningowych:
    - usuwa kolumny nadmiarowe
    - dodaje brakujące kolumny z zerami
    - ustawia tę samą kolejność kolumn
    """
    X_test_aligned = X_test.copy()

    train_cols = set(X_train.columns)
    test_cols = set(X_test.columns)

    # Kolumny brakujące w testowym – dodaj
    missing_cols = train_cols - test_cols
    for col in missing_cols:
        if col == 'const':
            X_test_aligned[col] = 1.0
        else:
            X_test_aligned[col] = 0

    # Kolumny nadmiarowe – usuń
    extra_cols = test_cols - train_cols
    X_test_aligned.drop(columns=extra_cols, inplace=True)

    # Kolejność kolumn
    X_test_aligned = X_test_aligned[X_train.columns]

    return X_test_aligned

def build_and_test_models(X_encoded, y_data, DUMMY_GROUPS, X_encoded_test, y_data_test):
    print("\n" + "="*60)
    print("ROZPOCZĘCIE BUDOWY I TESTOWANIA MODELI")
    print("="*60)
    

    current_model, X_data, y_data_clean, hellwig_vars, all_dummy_vars, all_vars = build_initial_model(X_encoded, y_data, DUMMY_GROUPS, False)

    diagnostic_results = run_diagnostic_tests(current_model, X_data, False)
    stability_results = test_model_stability(current_model, y_data_clean, X_data, False)
    run_test_t_student_significance(current_model, False)
    

    # METODA NAPRAWCZA 1: Usunięcie nieistotnych zmiennych na podstawie testy t-Studenta
    current_model, X_data, y_data_clean, X_data_test, y_data_clean_test = run_test_t_student_significance_and_remove(current_model, X_data, y_data_clean, build_model, X_encoded_test, y_data_test, False)

    # Testy po usunięciu
    diagnostic_results = run_diagnostic_tests(current_model, X_data, False)
    stability_results = test_model_stability(current_model, y_data_clean, X_data, False)

    # METODA NAPRAWCZA 2: Próba budowy modelu WMNK
    
    current_model, X_data, Y_data = build_weighted_model(X_data, y_data_clean, current_model, verbose=False)

    # Testy po usunięciu
    diagnostic_results = run_diagnostic_tests(current_model, X_data, False)
    stability_results = test_model_stability(current_model, y_data_clean, X_data, False)

    # METODA NAPRAWCZA 3: Transformacja logarytmiczna - w celu naprawy problemu z heteroskedastycznością -> nie naprawiły się, ale normalność się mocno naprawiła    
    current_model, X_data, y_log, X_data_test, y_log_test = logarithmic_transformation(X_data, y_data_clean, build_weighted_model, X_data_test, y_data_clean_test, False)
    
    # Testy po transformacji logarytmicznej
    diagnostic_results = run_diagnostic_tests(current_model, X_data, False)
    stability_results = test_model_stability(current_model, y_log, X_data, False)
    
    # METODA NAPRAWCZA 4:
    # Testy na normalność uległy znacznej poprawie, widzimy, że są 3 zmienne z VIF > 10, dlatego na początku usuwamy tylko jedną
    # Resztę zmiennych VIF > 10 pozostiwaimy poniewać:
    # model działa dobrze (R², testy reszt OK),
    # zmienna ma sens merytoryczny,
    X_data = X_data.drop(columns=["CPU_company_Intel"])
    X_data_test = X_data_test.drop(columns=["CPU_company_Intel"])
    
    diagnostic_results = run_diagnostic_tests(current_model, X_data, False)
    stability_results = test_model_stability(current_model, y_log, X_data, False)

    # plot_correlation_heatmap(X_data, target_variable="Price_euros")
    # # Usunięcie kolumny "GPU_company_Intel" ponieważ ma silną ujemną korelację (--0.72) z GPU_company_Nvidia
    # X_data = X_data.drop(columns=["GPU_company_Nvidia"])
    
    # METODA NAPRAWCZA 5: Przełamanie strukturalne    
    current_model, X_interactions, y_log = structural_break_correction(y_log, X_data, stability_results['break_point'], build_weighted_model, False)
    X_interactions_test = add_structural_break_columns(X_data_test, break_point=stability_results['break_point'], train_index_like=X_interactions.index)
    
    # # Testy po przełamaniu strukturalnym

    diagnostic_results = run_diagnostic_tests(current_model, X_interactions, False)
    stability_results = test_model_stability(current_model, y_log, X_interactions, False)
    
    # # METODA NAPRAWCZA 6: Korekta Ramsey RESET
    current_model, X_advanced, y_log, X_advanced_test, y_log_test = ramsey_reset_correction(X_interactions, y_log, build_weighted_model, X_interactions_test, y_log_test, verbose=False)
    # diagnostic_results = run_diagnostic_tests(current_model, X_advanced, False)
    # stability_results = test_model_stability(current_model, y_log, X_advanced, False)
    
    current_model, X_advanced, y_log = run_test_t_student_significance_and_remove_v2(current_model, X_advanced, y_log, build_weighted_model, False)
    current_model, X_advanced, y_log = run_test_t_student_significance_and_remove_v2(current_model, X_advanced, y_log, build_weighted_model, False)

    # # # Finalne testy

    run_test_t_student_significance(current_model, True)
    # diagnostic_results = run_diagnostic_tests(current_model, X_advanced, False)
    # stability_results = test_model_stability(current_model, y_log, X_advanced, False)
    
    # print(f"\n{'='*60}")
    # print("ZAKOŃCZENIE BUDOWY I TESTOWANIA MODELI")
    # print(f"{'='*60}")
    # print(f"Końcowy model R²: {current_model.rsquared:.4f}")
    # print(f"Końcowy model Adjusted R²: {current_model.rsquared_adj:.4f}")
    # print(f"Końcowy model AIC: {current_model.aic:.4f}")
    # print(f"Końcowy model BIC: {current_model.bic:.4f}")
    # X_advanced_test_aligned = align_test_columns(X_advanced, X_advanced_test)
    # compare_columns(X_advanced, X_advanced_test_aligned, label1="train", label2="test")
    
    # # df_results = calculate_forecast_errors(current_model, X_advanced, y_log)
    # df_results = calculate_forecast_errors(current_model, X_advanced_test_aligned, y_log_test)
    # print(f"FORECAST")
    # print(df_results)
    
    return current_model