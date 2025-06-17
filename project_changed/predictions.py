# from statsmodels.regression.linear_model import RegressionResultsWrapper
# import numpy as np
# import pandas as pd

# def calculate_forecast_errors(model: RegressionResultsWrapper, X, y_log, output_path=None):
#     """
#     Oblicza prognozę punktową i względny błąd prognozy EX POST (MAPE).
#     Zwraca DataFrame z y_true, y_pred, błędem bezwzględnym i względnym.

#     Returns:
#         df_results: DataFrame z wynikami.
#         mape: średni względny błąd prognozy w %.
#     """
#     # Predykcja logarytmiczna
#     y_pred_log = model.predict(X)

#     # Powrót do skali oryginalnej
#     y_pred = np.exp(y_pred_log)
#     y_true = np.exp(y_log)

#     # Błędy
#     error_abs = np.abs(y_true - y_pred)
#     error_rel = (error_abs / y_true) * 100
#     mape = np.mean(error_rel)

#     # Tabela wyników
#     df_results = pd.DataFrame({
#         'y_true': y_true,
#         'y_pred': y_pred,
#         'error_abs': error_abs,
#         'error_rel_%': error_rel
#     })

#     # Zapis do pliku
#     if output_path:
#         df_results.to_csv(output_path, index=False)

#     return df_results, mape
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.regression.linear_model import RegressionResultsWrapper

def calculate_forecast_errors(model: RegressionResultsWrapper, X, y_log):
    """
    Oblicza prognozy oraz błędy prognozy EX POST na danych logarytmicznych.
    Zwraca także obserwacje z największym i najmniejszym względnym błędem.
    """
    y_pred_log = model.predict(X)
    # Wymuszamy rzutowanie na float
    y_pred_log = pd.Series(y_pred_log).astype(float)
    
    y_pred = np.exp(y_pred_log)
    y_true = np.exp(y_log)

    error = y_true - y_pred
    error_abs = np.abs(error)
    error_rel_pct = 100 * error_abs / y_true

    # Metryki
    mape = np.mean(error_rel_pct)
    mae = mean_absolute_error(y_true, y_pred)
    me = np.mean(error)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Szukanie NAJMNIEJSZEGO i NAJWIĘKSZEGO błędu RELATYWNEGO
    max_error_idx = error_rel_pct.idxmax()
    min_error_idx = error_rel_pct.idxmin()

    # Tabela z wynikami
    df_results = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "error_abs": error_abs,
        "error_rel_%": error_rel_pct
    })

    # Drukowanie
    print(f"\n--- METRYKI BŁĘDÓW ---")
    print(f"MAPE: {mape:.2f}%")
    print(f"ME: {me:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"\nNajwiększy błąd względny ({error_rel_pct[max_error_idx]:.2f}%) dla obserwacji {max_error_idx}")
    print(df_results.loc[max_error_idx])
    print(f"\nNajmniejszy błąd względny ({error_rel_pct[min_error_idx]:.2f}%) dla obserwacji {min_error_idx}")
    print(df_results.loc[min_error_idx])

    return df_results, {
        "MAPE": mape,
        "ME": me,
        "MAE": mae,
        "RMSE": rmse,
        "max_error_index": int(max_error_idx),
        "min_error_index": int(min_error_idx),
        "max_error_%": float(error_rel_pct[max_error_idx]),
        "min_error_%": float(error_rel_pct[min_error_idx])
    }
