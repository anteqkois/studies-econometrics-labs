import pandas as pd
import numpy as np


def hellwig_method(X, y, dummy_groups=None, threshold=0.1, verbose=True):

    # 1. Obliczenie macierzy korelacji
    all_data = pd.concat([X, y], axis=1)
    corr_matrix = all_data.corr()
    
    # 2. Korelacje zmiennych objaśniających ze zmienną objaśnianą
    y_corr = corr_matrix.iloc[:-1, -1].abs()  # korelacje z Price_euros
    
    # 3. Eliminacja zmiennych kategorycznych
    dummy_vars = set()
    if dummy_groups:
        for group_vars in dummy_groups.values():
            dummy_vars.update(group_vars)
    y_corr = y_corr[~y_corr.index.isin(dummy_vars)]
    
    # 4. Eliminacja zmiennych o niskiej korelacji z y
    candidates = y_corr[y_corr >= threshold].index.tolist()
    if verbose:
        print(f"Zmienne kandydujące (korelacja >= {threshold}): {len(candidates)}")
    
    if len(candidates) == 0:
        if verbose:
            print("Brak zmiennych spełniających kryterium korelacji")
        return []
    
    # 5. Obliczenie wskaźnika pojemności informacyjnej dla każdej zmiennej
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
    
    # 6. Sortowanie według wskaźnika pojemności informacyjnej
    sorted_vars = sorted(capacity_indicators.items(), key=lambda x: x[1], reverse=True)
    
    if verbose:
        print("\nWskaźniki pojemności informacyjnej (Hellwig):")
        for var, h_i in sorted_vars[:10]:  # top 10
            print(f"{var}: {h_i:.4f}")
    
    # 7. Wybór zmiennych (można zastosować różne kryteria)
    # Tu wybieramy zmienne z wskaźnikiem > średnia
    mean_capacity = np.mean(list(capacity_indicators.values()))
    selected_vars = [var for var, h_i in sorted_vars if h_i > mean_capacity]
    
    if verbose:
        print(f"\nWybrane zmienne (wskaźnik > średnia = {mean_capacity:.4f}): {len(selected_vars)}")
        print("Wybrane zmienne:", selected_vars)
    
    return selected_vars

# Hellwig method which handle also groups
# def hellwig_method(X, y, dummy_groups=None, threshold=0.1, verbose=True):
#     all_data = pd.concat([X, y], axis=1)
#     corr_matrix = all_data.corr()
#     y_corr = corr_matrix.iloc[:-1, -1].abs()

#     if dummy_groups is None:
#         dummy_groups = {}

#     handled_vars = set()
#     block_candidates = {}
#     final_vars = []

#     # 1. Ocena bloków dummy
#     for group_name, cols in dummy_groups.items():
#         group_cols = [col for col in cols if col in X.columns]
#         if not group_cols:
#             continue

#         corr_vals = y_corr[group_cols]

#         if any(corr_vals >= threshold):
#             handled_vars.update(group_cols)

#             # Oblicz wskaźniki Hellwiga dla zmiennych w bloku
#             h_vals = {}
#             for var in group_cols:
#                 r_y = y_corr[var]
#                 others = [v for v in group_cols if v != var]
#                 r_x_sum = sum([corr_matrix.loc[var, other]**2 for other in others]) if others else 0
#                 h_i = r_y**2 / (1 + r_x_sum) if r_x_sum < 1 else 0
#                 h_vals[var] = h_i

#             mean_block_hi = np.mean(list(h_vals.values()))
#             block_candidates[group_name] = (group_cols, mean_block_hi)

#     # 2. Pozostałe zmienne
#     remaining_vars = [var for var in y_corr.index if var not in handled_vars and y_corr[var] >= threshold]
#     single_var_hi = {}

#     for var in remaining_vars:
#         r_y = y_corr[var]
#         others = [v for v in remaining_vars if v != var]
#         r_x_sum = sum([corr_matrix.loc[var, other]**2 for other in others]) if others else 0
#         h_i = r_y**2 / (1 + r_x_sum) if r_x_sum < 1 else 0
#         single_var_hi[var] = h_i

#     # 3. Ustal próg na podstawie wszystkich h_i
#     all_hi_values = list(single_var_hi.values()) + [v[1] for v in block_candidates.values()]
#     mean_hi = np.mean(all_hi_values)

#     if verbose:
#         print(f"\nŚredni wskaźnik pojemności informacyjnej: {mean_hi:.4f}")

#     # 4. Wybór zmiennych i bloków
#     for var, h_i in single_var_hi.items():
#         if h_i > mean_hi:
#             final_vars.append(var)

#     for group_name, (cols, h_i) in block_candidates.items():
#         if h_i > mean_hi:
#             final_vars.extend(cols)
#             if verbose:
#                 print(f"[HELLWIG] Blok '{group_name}' dodany (średni h_i = {h_i:.4f})")

#     if verbose:
#         print(f"\nWybrane zmienne (łącznie): {len(final_vars)}")
#         print("Wybrane zmienne:", final_vars)

#     return final_vars
