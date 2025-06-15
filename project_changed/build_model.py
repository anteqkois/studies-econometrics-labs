from typing import Optional
from statsmodels.regression.linear_model import RegressionResultsWrapper
import statsmodels.api as sm
import re
from pandas import DataFrame

def extract_model_stats(summary_text):
    """Wyciąga statystyki DW i JB z podsumowania tekstowego."""
    dw_match = re.search(r'Durbin-Watson:\s+([0-9.]+)', summary_text)
    jb_match = re.search(r'Jarque-Bera \(JB\):\s+([0-9.]+)', summary_text)
    jb_pval_match = re.search(r'Prob\(JB\):\s+([0-9.eE-]+)', summary_text)

    dw_stat = float(dw_match.group(1)) if dw_match else None
    jb_stat = float(jb_match.group(1)) if jb_match else None
    jb_pval = float(jb_pval_match.group(1)) if jb_pval_match else None

    return dw_stat, jb_stat, jb_pval


def build_model(X_data, y_data_clean, verbose=True):
    X_data_with_const = sm.add_constant(X_data)
    model = sm.OLS(y_data_clean, X_data_with_const).fit()

    if verbose:
        print("="*60)
        print(f"\nModel MNK z {X_data.shape[1]} zmiennymi")

        # Wzór
        params = model.params
        terms = [f"{coef:.4f} * {name}" if name != "const" else f"{coef:.4f}"
                 for name, coef in params.items()]
        model_equation = " + ".join(terms)
        print("\nWzór modelu regresji liniowej (MNK):")
        print(f"Y = {model_equation}")

        # Statystyki z podsumowania
        summary_text = model.summary().as_text()
        dw_stat, jb_stat, jb_pval = extract_model_stats(summary_text)

        print("\nStatystyki modelu:")
        print(f"R²: {model.rsquared:.4f}")
        print(f"Durbin-Watson: {dw_stat:.4f}" if dw_stat else "Durbin-Watson: brak")
        print(f"Jarque-Bera (JB): {jb_stat:.4f}, p-value: {jb_pval:.4f}" if jb_stat else "JB: brak")
        print("="*60)

    return model, X_data_with_const, y_data_clean


def build_weighted_model(X_data, y_data_clean, ols_model: Optional[RegressionResultsWrapper] = None, weights=None, verbose=True):
    X_data_with_const = sm.add_constant(X_data)
    X_data_with_const = DataFrame(X_data_with_const)  # jawne rzutowanie

    if weights is None:
        if ols_model is None:
            ols_model = sm.OLS(y_data_clean, X_data_with_const).fit()
        resid_squared = ols_model.resid ** 2
        weights = 1 / (resid_squared + 1e-6)
        print("\nSTATYSTYKI WAG:")
        print(f"Min: {weights.min():.6f}, Max: {weights.max():.6f}, Mean: {weights.mean():.6f}, Std: {weights.std():.6f}")
    
    if ols_model is None:
            ols_model = sm.OLS(y_data_clean, X_data_with_const).fit()

    model = sm.WLS(y_data_clean, X_data_with_const, weights=weights).fit()

    if verbose:
        print("="*60)
        print(f"\nModel ważony (WMNK) z {X_data.shape[1]} zmiennymi")

        # Wzór
        params_wls = model.params
        terms_wls = [f"{coef:.4f} * {name}" if name != "const" else f"{coef:.4f}"
                 for name, coef in params_wls.items()]
        model_equation_wls = " + ".join(terms_wls)
        print("\nWzór modelu regresji liniowej (WMNK):")
        print(f"Y = {model_equation_wls}")
        
        params_ols = ols_model.params
        terms_ols = [f"{coef:.4f} * {name}" if name != "const" else f"{coef:.4f}"
                 for name, coef in params_ols.items()]
        model_equation_ols = " + ".join(terms_ols)
        print("\nWzór modelu regresji liniowej (MNK):")
        print(f"Y = {model_equation_ols}")

        # Statystyki z podsumowania
        summary_text_wls = model.summary().as_text()
        dw_stat_wls, jb_stat_wls, jb_pval_wls = extract_model_stats(summary_text_wls)
        
        summary_text_ols = ols_model.summary().as_text()
        dw_stat_ols, jb_stat_ols, jb_pval_ols = extract_model_stats(summary_text_ols)

        print("\nStatystyki modelu:")
        print(f"R²: {model.rsquared:.4f} (OLS: {ols_model.rsquared:.4f})")
        print(f"Durbin-Watson: {dw_stat_wls:.4f} (OLS: {dw_stat_ols:.4f})" if dw_stat_wls else "Durbin-Watson: brak")
        print(f"Jarque-Bera (JB): {jb_stat_wls:.4f} (OLS: {jb_stat_ols:.4f}), p-value: {jb_pval_wls:.4f} (OLS: {jb_pval_ols:.4f})" if jb_stat_wls else "JB: brak")
        print("="*60)

    return model, X_data_with_const, y_data_clean
