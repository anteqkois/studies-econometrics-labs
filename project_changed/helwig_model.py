import pandas as pd
import numpy as np


def hellwig_method(X, y, threshold=0.1, verbose=True):

    # 1. Obliczenie macierzy korelacji
    all_data = pd.concat([X, y], axis=1)
    corr_matrix = all_data.corr()
    
    # 2. Korelacje zmiennych objaśniających ze zmienną objaśnianą
    y_corr = corr_matrix.iloc[:-1, -1].abs()  # korelacje z Price_euros
    
    # 3. Eliminacja zmiennych o niskiej korelacji z y
    candidates = y_corr[y_corr >= threshold].index.tolist()
    if verbose:
        print(f"Zmienne kandydujące (korelacja >= {threshold}): {len(candidates)}")
    
    if len(candidates) == 0:
        if verbose:
            print("Brak zmiennych spełniających kryterium korelacji")
        return []
    
    # 4. Obliczenie wskaźnika pojemności informacyjnej dla każdej zmiennej
    capacity_indicators = {}
    
    for var in candidates:
        r_y = y_corr[var]  # korelacja ze zmienną objaśnianą
        
        # Suma kwadratów korelacji z pozostałymi zmiennymi objaśniającymi
        other_vars = [v for v in candidates if v != var]
        if len(other_vars) > 0:
            r_x_sum = sum([corr_matrix.loc[var, other_var]**2 for other_var in other_vars])
        else:
            r_x_sum = 0
        
        # Wskaźnik pojemności informacyjnej
        if r_x_sum < 1:
            h_i = r_y**2 / (1 + r_x_sum)
        else:
            h_i = 0  # bardzo wysoka współliniowość
        
        capacity_indicators[var] = h_i
    
    # 5. Sortowanie według wskaźnika pojemności informacyjnej
    sorted_vars = sorted(capacity_indicators.items(), key=lambda x: x[1], reverse=True)
    
    if verbose:
        print("\nWskaźniki pojemności informacyjnej (Hellwig):")
        for var, h_i in sorted_vars[:10]:  # top 10
            print(f"{var}: {h_i:.4f}")
    
    # 6. Wybór zmiennych (można zastosować różne kryteria)
    # Tu wybieramy zmienne z wskaźnikiem > średnia
    mean_capacity = np.mean(list(capacity_indicators.values()))
    selected_vars = [var for var, h_i in sorted_vars if h_i > mean_capacity]
    
    if verbose:
        print(f"\nWybrane zmienne (wskaźnik > średnia = {mean_capacity:.4f}): {len(selected_vars)}")
        print("Wybrane zmienne:", selected_vars)
    
    return selected_vars