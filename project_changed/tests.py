from scipy import stats
import statsmodels.stats.diagnostic as smd
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


def test_normality(residuals, alpha=0.05, verbose=True):

    results = {}
    
    if len(residuals) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        if verbose:
            print(f"Test Shapiro-Wilka:")
            print(f"  Statystyka: {shapiro_stat:.4f}")
            print(f"  P-value: {shapiro_p:.4f}")
        
        if shapiro_p > alpha:
            interpretation = f"Nie ma podstaw do odrzucenia hipotezy o normalności (α={alpha})"
        else:
            interpretation = f"Odrzucamy hipotezę o normalności (α={alpha})"
        if verbose:
            print(f"{interpretation}")
        
        results['shapiro'] = {
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'interpretation': interpretation
        }
    else:
        if verbose:
            print("Test Shapiro-Wilka: Próbka zbyt duża (n > 5000)")
        results['shapiro'] = None

    # Test Kołmogorowa-Smirnowa
    ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(residuals.mean(), residuals.std()))
    if verbose:
        print(f"\nTest Kołmogorowa-Smirnowa:")
        print(f"  Statystyka: {ks_stat:.4f}")
        print(f"  P-value: {ks_p:.4f}")
    
    if ks_p > alpha:
        ks_interpretation = f"Nie ma podstaw do odrzucenia hipotezy o normalności (α={alpha})"
    else:
        ks_interpretation = f"Odrzucamy hipotezę o normalności (α={alpha})"
    if verbose:
        print(f"  → {ks_interpretation}")
    
    results['kolmogorov_smirnov'] = {
        'statistic': ks_stat,
        'p_value': ks_p,
        'interpretation': ks_interpretation
    }

    # Test Jarque-Bera (dobry dla dużych próbek)
    jb_stat, jb_p = stats.jarque_bera(residuals)
    if verbose:
        print(f"\nTest Jarque-Bera:")
        print(f"Statystyka: {jb_stat:.4f}")
        print(f"P-value: {jb_p:.4f}")
    
    if jb_p > alpha:
        jb_interpretation = f"Nie ma podstaw do odrzucenia hipotezy o normalności (α={alpha})"
    else:
        jb_interpretation = f"Odrzucamy hipotezę o normalności (α={alpha})"
    if verbose:
        print(f"{jb_interpretation}")
    
    results['jarque_bera'] = {
        'statistic': jb_stat,
        'p_value': jb_p,
        'interpretation': jb_interpretation
    }
    
    return results


def test_heteroscedasticity(residuals, X_hellwig_with_const, verbose=True):

    bp_lm, bp_lm_pvalue, bp_fvalue, bp_f_pvalue = smd.het_breuschpagan(residuals, X_hellwig_with_const)
    if verbose:
        print(f"Test Breuscha-Pagana:")
        print(f"Statystyka LM: {bp_lm:.4f}")
        print(f"P-value (LM): {bp_lm_pvalue:.4f}")
        print(f"Statystyka F: {bp_fvalue:.4f}")
        print(f"P-value (F): {bp_f_pvalue:.4f}")
        if bp_lm_pvalue > 0.05:
            print("Nie ma podstaw do odrzucenia hipotezy o homoskedastyczności (α=0.05)")
            print("Wariancja reszt jest stała (homoskedastyczność)")
        else:
            print("Odrzucamy hipotezę o homoskedastyczności (α=0.05)")
            print("Występuje heteroskedastyczność")

    # 2. Test White'a
    white_lm, white_lm_pvalue, white_fvalue, white_f_pvalue = smd.het_white(residuals, X_hellwig_with_const)
    if verbose:
        print(f"\nTest White'a:")
        print(f"  Statystyka LM: {white_lm:.4f}")
        print(f"  P-value (LM): {white_lm_pvalue:.4f}")
        print(f"  Statystyka F: {white_fvalue:.4f}")
        print(f"  P-value (F): {white_f_pvalue:.4f}")
        if white_lm_pvalue > 0.05:
            print("Nie ma podstaw do odrzucenia hipotezy o homoskedastyczności (α=0.05)")
            print("Wariancja reszt jest stała (homoskedastyczność)")
        else:
            print("Odrzucamy hipotezę o homoskedastyczności (α=0.05)")
            print("Występuje heteroskedastyczność")


def test_vif(X, threshold=10.0, verbose=True):
    if verbose:
        print(f"\n{'='*40}")
        print("TEST WSPÓŁLINIOWOŚCI (VIF)")
        print(f"{'='*40}")
    
    # Konwersja do DataFrame jeśli potrzeba
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # Obliczenie VIF dla każdej zmiennej
    vif_data = pd.DataFrame()
    vif_data["Zmienna"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    # Sortowanie według VIF (malejąco)
    vif_data = vif_data.sort_values('VIF', ascending=False)
    
    if verbose:
        print(f"Próg VIF: {threshold}")
        print(f"Zmienne z VIF > {threshold} wskazują na problem współliniowości\n")
        
        # Wyświetlenie wyników
        print("Wartości VIF:")
        print("-" * 30)
        for _, row in vif_data.iterrows():
            status = "PROBLEM" if row['VIF'] > threshold else "✓ OK"
            inf_status = " (Infinity)" if np.isinf(row['VIF']) else ""
            print(f"{row['Zmienna']:25s}: {row['VIF']:8.2f}{inf_status} {status}")
        
        # Interpretacja
        print(f"\n")
        print("INTERPRETACJA VIF:")

        print("VIF = 1: Brak współliniowości")
        print("1 < VIF < 5: Umiarkowana współliniowość")
        print("5 ≤ VIF < 10: Wysoka współliniowość") 
        print("VIF ≥ 10: Bardzo wysoka współliniowość (problem)")
        print("VIF = infinity: Doskonała współliniowość")
        
        # Podsumowanie
        problematic_vars = vif_data[vif_data['VIF'] > threshold]
        if len(problematic_vars) > 0:
            print(f"\nOSTRZEŻENIE: Znaleziono {len(problematic_vars)} zmiennych z VIF > {threshold}:")
        else:
            print(f"\nWszystkie zmienne mają VIF ≤ {threshold} - brak problemów ze współliniowością")
    
    return vif_data


def test_chow(y, X, break_point, alpha=0.05, verbose=True):

    if verbose:
        print(f"\n{'='*40}")
        print("TEST CHOWA - STABILNOŚĆ PARAMETRÓW")
        print(f"{'='*40}")
    
    # Konwersja do odpowiednich typów
    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(X, pd.DataFrame):
        X = X.values

    
    n = len(y)
    k = X.shape[1]  # liczba parametrów (włącznie ze stałą)
    

    if verbose:
        print(f"Pierwsza podprόba: obserwacje 0-{break_point-1} ({break_point} obs.)")
        print(f"Druga podprόba: obserwacje {break_point}-{n-1} ({n-break_point} obs.)")
    
    # Sprawdzenie czy podpróby mają wystarczającą liczbę obserwacji
    if break_point < k or (n - break_point) < k:
        if verbose:
            print(f"\nBŁĄD: Podpróby muszą mieć co najmniej {k} obserwacji każda")
            print(f"Pierwsza podpróba: {break_point} obserwacji")
            print(f"Druga podpróba: {n - break_point} obserwacji")
        return None
    
    # Podział danych
    y1, y2 = y[:break_point], y[break_point:]
    X1, X2 = X[:break_point], X[break_point:]
    
    # Model dla całej próby
    model_full = sm.OLS(y, X).fit()
    rss_full = np.sum(model_full.resid**2)
    
    # Modele dla podprób
    model_1 = sm.OLS(y1, X1).fit()
    model_2 = sm.OLS(y2, X2).fit()
    
    rss_1 = np.sum(model_1.resid**2)
    rss_2 = np.sum(model_2.resid**2)
    rss_separate = rss_1 + rss_2
    
    # Statystyka testu Chowa
    f_stat = ((rss_full - rss_separate) / k) / (rss_separate / (n - 2*k))
    
    # Wartość krytyczna i p-value
    f_critical = stats.f.ppf(1 - alpha, k, n - 2*k)
    p_value = 1 - stats.f.cdf(f_stat, k, n - 2*k)
    
    if verbose:
        print(f"\n")
        print("WYNIKI TESTU CHOWA:")
        print(f"RSS (model pełny): {rss_full:.4f}")
        print(f"RSS (podpróba 1): {rss_1:.4f}")
        print(f"RSS (podpróba 2): {rss_2:.4f}")
        print(f"RSS (podpróby razem): {rss_separate:.4f}")
        print(f"\nStatystyka F: {f_stat:.4f}")
        print(f"Wartość krytyczna F({k}, {n-2*k}) przy α={alpha}: {f_critical:.4f}")
        print(f"P-value: {p_value:.4f}")
        

        print(f"\n")
        print("INTERPRETACJA:")

        print("H₀: Parametry są stabilne w obu podpróbach")
        print("H₁: Parametry różnią się między podpróbami")
    
    if p_value < alpha:
        conclusion = f"Odrzucamy H₀ przy α={alpha}"
        interpretation = "Parametry modelu NIE SĄ stabilne - występuje przełamanie strukturalne"
        status = "NIESTABILNE"
    else:
        conclusion = f"Nie ma podstaw do odrzucenia H₀ przy α={alpha}"
        interpretation = "Parametry modelu SĄ stabilne - brak przełamania strukturalnego"
        status = "STABILNE"
    
    if verbose:
        print(f"\n{conclusion}")
        print(f"{interpretation}")
        print(f"\nStatus: {status}")
    

    results = {
        'f_statistic': f_stat,
        'f_critical': f_critical,
        'p_value': p_value,
        'rss_full': rss_full,
        'rss_separate': rss_separate,
        'conclusion': conclusion,
        'interpretation': interpretation,
        'stable': p_value >= alpha
    }
    
    return results


def test_ramsey_reset(model, alpha=0.05, power=2, verbose=True):

    if verbose:
        print(f"\n{'='*40}")
        print("TEST RAMSEY'A RESET")
        print(f"{'='*40}")
    
    y = model.model.endog
    X = model.model.exog
    fitted_values = model.fittedvalues
    n = len(y)
    k = X.shape[1]  # liczba parametrów w oryginalnym modelu
        
    # Tworzenie potęg dopasowanych wartości
    X_extended = X.copy()
    added_vars = []
    
    for p in range(2, power + 1):
        var_name = f"fitted_power_{p}"
        fitted_power = fitted_values ** p
        X_extended = np.column_stack([X_extended, fitted_power])
        added_vars.append(var_name)
    
    num_added = len(added_vars)
    if verbose:
        print(f"Liczba dodanych zmiennych: {num_added}")
    
    # Model rozszerzony
    try:
        extended_model = sm.OLS(y, X_extended).fit()
        
        # RSS dla modelu ograniczonego (oryginalnego) i nieograniczonego (rozszerzonego)
        rss_restricted = np.sum(model.resid**2)
        rss_unrestricted = np.sum(extended_model.resid**2)
        
        # Statystyka F dla testu RESET
        f_stat = ((rss_restricted - rss_unrestricted) / num_added) / (rss_unrestricted / (n - k - num_added))
        
        # Wartość krytyczna i p-value
        f_critical = stats.f.ppf(1 - alpha, num_added, n - k - num_added)
        p_value = 1 - stats.f.cdf(f_stat, num_added, n - k - num_added)
        
        # Wyniki
        if verbose:
            print(f"\n")
            print("WYNIKI TESTU RESET:")
            print(f"RSS (model ograniczony): {rss_restricted:.4f}")
            print(f"RSS (model nieograniczony): {rss_unrestricted:.4f}")
            print(f"Różnica RSS: {rss_restricted - rss_unrestricted:.4f}")
            print(f"\nStatystyka F: {f_stat:.4f}")
            print(f"Wartość krytyczna F({num_added}, {n-k-num_added}) przy α={alpha}: {f_critical:.4f}")
            print(f"P-value: {p_value:.4f}")
        
        # Interpretacja
        if verbose:
            print(f"\n")
            print("INTERPRETACJA:")
            print("H₀: Postać analityczna modelu jest prawidłowa")
            print("H₁: Postać analityczna modelu jest nieprawidłowa (brakuje nieliniowych składników)")
        
        if p_value < alpha:
            conclusion = f"Odrzucamy H₀ przy α={alpha}"
            interpretation = "Postać analityczna modelu jest NIEPRAWIDŁOWA"
            status = "NIEPRAWIDŁOWA"
        else:
            conclusion = f"Nie ma podstaw do odrzucenia H₀ przy α={alpha}"
            interpretation = "Postać analityczna modelu jest PRAWIDŁOWA"
            status = "PRAWIDŁOWA"
        
        if verbose:
            print(f"\n{conclusion}")
            print(f"{interpretation}")
            print(f"\nStatus: {status}")
        
     
        results = {
            'f_statistic': f_stat,
            'f_critical': f_critical,
            'p_value': p_value,
            'rss_restricted': rss_restricted,
            'rss_unrestricted': rss_unrestricted,
            'conclusion': conclusion,
            'interpretation': interpretation,
            'status': status,
            'correctly_specified': p_value >= alpha
        }
        
        return results
        
    except Exception as e:
        if verbose:
            print(f"Błąd podczas wykonywania testu RESET: {e}")
        return None


def test_runs(residuals, alpha=0.05, verbose=True):

    if verbose:
        print(f"\n{'='*40}")
        print("TEST LICZBY SERII (RUNS TEST)")
        print(f"{'='*40}")
    
    # Konwersja do array
    if hasattr(residuals, 'values'):
        residuals = residuals.values
    
    n = len(residuals)
    if verbose:
        print(f"Liczba obserwacji: {n}")
    
    # Konwersja reszt na sekwencję binarną względem mediany
    median_resid = np.median(residuals)
    binary_sequence = (residuals > median_resid).astype(int)
    
    # Liczenie dodatnich i ujemnych reszt
    n_positive = np.sum(binary_sequence == 1)
    n_negative = np.sum(binary_sequence == 0)
    
    if verbose:
        print(f"Mediana reszt: {median_resid:.4f}")
        print(f"Liczba reszt > mediany: {n_positive}")
        print(f"Liczba reszt ≤ mediany: {n_negative}")
    
    # Sprawdzenie czy test jest możliwy
    if n_positive == 0 or n_negative == 0:
        if verbose:
            print("Błąd: Wszystkie reszty mają ten sam znak względem mediany")
        return None
    
    # Liczenie serii
    runs = 1
    for i in range(1, n):
        if binary_sequence[i] != binary_sequence[i-1]:
            runs += 1
    
    if verbose:
        print(f"Liczba serii: {runs}")
    
    # Oczekiwana liczba serii i wariancja (dla dużych próbek)
    expected_runs = (2 * n_positive * n_negative) / n + 1
    variance_runs = (2 * n_positive * n_negative * (2 * n_positive * n_negative - n)) / (n**2 * (n - 1))
    
    if verbose:
        print(f"Oczekiwana liczba serii: {expected_runs:.4f}")
        print(f"Wariancja liczby serii: {variance_runs:.4f}")
    
    if variance_runs <= 0:
        if verbose:
            print("BŁĄD: Wariancja liczby serii ≤ 0")
        return None
    
    std_runs = np.sqrt(variance_runs)
    if verbose:
        print(f"Odchylenie standardowe: {std_runs:.4f}")
    
    # Statystyka testowa z poprawką na ciągłość
    if runs > expected_runs:
        z_stat = (runs - 0.5 - expected_runs) / std_runs
    else:
        z_stat = (runs + 0.5 - expected_runs) / std_runs
    
    # P-value (test dwustronny)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # Wartość krytyczna
    z_critical = stats.norm.ppf(1 - alpha/2)
    
    # Wyniki
    if verbose:
        print(f"\n")
        print("WYNIKI TESTU LICZBY SERII:")

        
        print(f"Statystyka Z: {z_stat:.4f}")
        print(f"Wartość krytyczna Z przy α={alpha}: ±{z_critical:.4f}")
        print(f"P-value (test dwustronny): {p_value:.4f}")
        
        # Interpretacja
        print(f"\n")
        print("INTERPRETACJA:")
        print("H₀: Reszty są rozmieszczone losowo")
        print("H₁: Reszty wykazują systematyczny wzorzec")
    
    if p_value < alpha:
        conclusion = f"Odrzucamy H₀ przy α={alpha}"
        if runs < expected_runs:
            interpretation = "Reszty wykazują KLASTROWANIE (zbyt mało serii)"
            pattern = "klastrowanie"
        else:
            interpretation = "Reszty wykazują OSCYLACJE (zbyt dużo serii)"
            pattern = "oscylacje"
        status = "NIELOSOWE"
        recommendation = "Model może wymagać dodatkowych zmiennych lub innej specyfikacji"
    else:
        conclusion = f"Nie ma podstaw do odrzucenia H₀ przy α={alpha}"
        interpretation = "Reszty są rozmieszczone LOSOWO"
        pattern = "losowe"
        status = "LOSOWE"
        recommendation = "Rozkład reszt jest zadowalający"
    
    if verbose:
        print(f"\n{conclusion}")
        print(f"{interpretation}")
        print(f"\nStatus: {status}")
        print(f"Wzorzec: {pattern}")
        print(f"Wynik: {recommendation}")
    

    results = {
        'z_statistic': z_stat,
        'z_critical': z_critical,
        'p_value': p_value,
        'runs_observed': runs,
        'runs_expected': expected_runs,
        'n_positive': n_positive,
        'n_negative': n_negative,
        'conclusion': conclusion,
        'interpretation': interpretation,
        'status': status,
        'pattern': pattern,
        'recommendation': recommendation,
        'random': p_value >= alpha
    }
    
    return results


def test_coincidence(X, threshold=0.8, alpha=0.05):
    print(f"\n{'='*60}")
    print("TEST KOINCYDENCJI - WYSOKIE KORELACJE MIĘDZY ZMIENNYMI")
    print(f"{'='*60}")
    
    # Konwersja do DataFrame jeśli potrzeba
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    print(f"Liczba zmiennych objaśniających: {X.shape[1]}")
    print(f"Próg korelacji: {threshold}")
    print(f"Poziom istotności: {alpha}")
    
    # Obliczenie macierzy korelacji
    corr_matrix = X.corr()
    
    # Znalezienie par zmiennych z wysoką korelacją
    high_corr_pairs = []
    problematic_vars = set()
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            var1 = corr_matrix.columns[i]
            var2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            
            if abs(corr_value) >= threshold:
                # Test istotności korelacji
                n = len(X)
                t_stat = corr_value * np.sqrt((n-2)/(1-corr_value**2))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
                
                high_corr_pairs.append({
                    'var1': var1,
                    'var2': var2,
                    'correlation': corr_value,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < alpha
                })
                
                problematic_vars.add(var1)
                problematic_vars.add(var2)
    
    # Wyświetlenie wyników
    print(f"\n{'='*40}")
    print("PARY ZMIENNYCH Z WYSOKĄ KORELACJĄ:")
    print(f"{'='*40}")
    
    if len(high_corr_pairs) == 0:
        print(f"Nie znaleziono par zmiennych z korelacją |r| ≥ {threshold}")
    else:
        print(f"Znaleziono {len(high_corr_pairs)} par zmiennych z wysoką korelacją:")
        print("\nSzczegóły:")
        print(f"{'Zmienna 1':<20} {'Zmienna 2':<20} {'Korelacja':<12} {'P-value':<12} {'Istotna'}")
        print("-" * 80)
        
        for pair in high_corr_pairs:
            significance = "***" if pair['p_value'] < 0.001 else "**" if pair['p_value'] < 0.01 else "*" if pair['p_value'] < 0.05 else ""
            istotna = "TAK" if pair['significant'] else "NIE"
            print(f"{pair['var1']:<20} {pair['var2']:<20} {pair['correlation']:>10.4f} {pair['p_value']:>10.4f} {istotna} {significance}")
    
    # Analiza problematycznych zmiennych
    print(f"\n{'='*40}")
    print("ANALIZA PROBLEMATYCZNYCH ZMIENNYCH:")
    print(f"{'='*40}")
    
    if len(problematic_vars) == 0:
        print("Brak zmiennych z problemem koincydencji")
        overall_status = "BRAK PROBLEMÓW"
    else:
        print(f"Zmienne z problemem koincydencji ({len(problematic_vars)}):")
        
        # Zliczenie wystąpień każdej zmiennej w parach problematycznych
        var_counts = {}
        for pair in high_corr_pairs:
            var_counts[pair['var1']] = var_counts.get(pair['var1'], 0) + 1
            var_counts[pair['var2']] = var_counts.get(pair['var2'], 0) + 1
        
        # Sortowanie według liczby wystąpień
        sorted_vars = sorted(var_counts.items(), key=lambda x: x[1], reverse=True)
        
        for var, count in sorted_vars:
            print(f"  {var}: {count} problematycznych korelacji")
        
        overall_status = "PROBLEMY WYKRYTE"
    
    # Macierz korelacji - najwyższe wartości
    print(f"\n{'='*40}")
    print("NAJWYŻSZE KORELACJE (TOP 10):")
    print(f"{'='*40}")
    
    # Utworzenie listy wszystkich korelacji (bez duplikatów i autokorelacji)
    all_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            all_correlations.append({
                'var1': corr_matrix.columns[i],
                'var2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })
    
    # Sortowanie według wartości bezwzględnej korelacji
    all_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    print(f"{'Zmienna 1':<20} {'Zmienna 2':<20} {'Korelacja':<12}")
    print("-" * 55)
    for corr in all_correlations[:10]:
        print(f"{corr['var1']:<20} {corr['var2']:<20} {corr['correlation']:>10.4f}")
    
    # Rekomendacje
    print(f"\n{'='*40}")
    print("REKOMENDACJE:")
    print(f"{'='*40}")
    
    if len(problematic_vars) == 0:
        print("✓ Brak problemów z koincydencją zmiennych")
        print("✓ Można kontynuować analizę bez dodatkowych działań")
        recommendations = ["Brak działań naprawczych"]
    else:
        print("⚠ Wykryto problemy z koincydencją zmiennych")
        recommendations = []
        
        if len(problematic_vars) > 0:
            print("\nMożliwe działania naprawcze:")
            print("1. Usunięcie jednej ze zmiennych w każdej problematycznej parze")
            print("2. Kombinacja skorelowanych zmiennych (np. średnia, suma)")
            print("3. Analiza głównych składowych (PCA)")
            print("4. Regularyzacja (Ridge, Lasso)")
            print("5. Sprawdzenie VIF dla potwierdzenia problemów")
            
            recommendations = [
                "Rozważ usunięcie najbardziej problematycznych zmiennych",
                "Przeprowadź analizę VIF",
                "Rozważ zastosowanie PCA lub regularyzacji"
            ]
        
        # Najbardziej problematyczna zmienna
        if len(var_counts) > 0:
            most_problematic = max(var_counts.items(), key=lambda x: x[1])
            print(f"\nNajbardziej problematyczna zmienna: {most_problematic[0]} ({most_problematic[1]} korelacji)")
            recommendations.append(f"Rozważ usunięcie zmiennej: {most_problematic[0]}")
    
    # Podsumowanie wyników
    results = {
        'high_correlation_pairs': high_corr_pairs,
        'problematic_variables': list(problematic_vars),
        'num_problematic_pairs': len(high_corr_pairs),
        'num_problematic_variables': len(problematic_vars),
        'correlation_matrix': corr_matrix,
        'threshold': threshold,
        'overall_status': overall_status,
        'recommendations': recommendations,
        'has_problems': len(problematic_vars) > 0
    }
    
    print(f"\n{'='*40}")
    print("PODSUMOWANIE:")
    print(f"{'='*40}")
    print(f"Status: {overall_status}")
    print(f"Liczba problematycznych par: {len(high_corr_pairs)}")
    print(f"Liczba problematycznych zmiennych: {len(problematic_vars)}")
    
    return results

