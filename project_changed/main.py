import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.stats import skew
from pandas.api.types import is_numeric_dtype
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
import statsmodels.stats.diagnostic as smd


# Wczytanie danych z pliku
df = pd.read_csv('kois_mazur_projekt_train_data.csv')

# Wyświetlenie pierwszych kilku wierszy danych
# print(df.head())

# Sprawdzanie brakujących danych
# print(df.isnull().sum())


# Podstawowe statystyki opisowe

# Wybieramy wszystkie kolumny oprócz 'id'
numeric_cols = df.select_dtypes(include='number').columns

stats = pd.DataFrame({
    'mean':      df[numeric_cols].mean(),
    'std':       df[numeric_cols].std(ddof=1),   # odchylenie standardowe (próba)
    'kurtosis':  df[numeric_cols].kurt(),        # kurtoza (nadmiarowa, 0 = N(0,1))
    'skewness':  df[numeric_cols].skew()         # skośność
})

# Dodanie współczynnika zmienności
stats['coef_var'] = stats['std'] / stats['mean']

# print(stats)


# Macierz korelacji Pearsona
corr = df[numeric_cols].corr(method="pearson")

# print("\nMacierz korelacji (zaokrąglenie do 5 miejsc):\n")
# print(corr.round(5))

# Wizualizacja korelacji – mapa cieplna
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, vmin=-1, vmax=1)  # default colormap

# Opisy osi
ax.set_xticks(range(len(numeric_cols)))
ax.set_xticklabels(numeric_cols, rotation=90, ha="center")
ax.set_yticks(range(len(numeric_cols)))
ax.set_yticklabels(numeric_cols)

# Dodanie wartości liczbowych do komórek
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        value = corr.iloc[i, j]
        ax.text(j, i, f"{value:.2f}", ha='center', va='center', color='black')

# Dodanie czerwonej ramki wokół wiersza "Price_euros"
if "Price_euros" in numeric_cols:
    row_index = list(numeric_cols).index("Price_euros")
    rect = patches.Rectangle(
        (-0.5, row_index - 0.5),
        len(numeric_cols),
        1,
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)

# Pasek kolorów
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Współczynnik korelacji", rotation=90)

plt.title("Macierz korelacji – heatmap")
plt.tight_layout()
# plt.show()



# kolumny do całkowitego usunięcia
DROP_COLS = [
    'Product',
    'TypeName',
    'CPU_model',
    'ScreenH',
    'GPU_model',
    'SecondaryStorageType'
]

# zostawiamy TOP_K a resztę ozancxamy 'Other'
CAT_TOPK_COLS = [
    'Company',
    'OS',
]

# mało kategorii,  zostawiamy wszystkie
CAT_SMALL_COLS = [
    'Screen',
    'CPU_company',
    'PrimaryStorageType',
    'GPU_company'
]

# zmienne 'Yes'/'Not'
BINARY_MAP = {'Yes': 1, 'No': 0}
BIN_COLS = ['Touchscreen', 'IPSpanel', 'RetinaDisplay']


# wszystkie zmienne liczbowe, które zachowujemy
NUM_COLS = [
    'Inches', 'Ram', 'Weight',
    'ScreenW', 'CPU_freq',
    'PrimaryStorage', 'SecondaryStorage'
]

for col in CAT_TOPK_COLS:
    n = df[col].nunique()
    # print(f"{col}: {n} unikalnych kategorii")
    
# ile najczęstszych kategorii zostawić
TOP_K = {
    'Company': 6,
    'OS': 5
}


# Usuwamy zbędne kolumny
df = df.drop(columns=DROP_COLS, errors='ignore')


# Redukujemy zmienne kategoryczne tak by zawierały TOP_K kategorii a reszta została zmieniona na 'Other', czyli pozbywamy się wartości odstających w zmiennych kategorycznych
for col in CAT_TOPK_COLS:
    vc = df[col].value_counts()          # liczebność każdej kategorii
    top = vc.nlargest(TOP_K[col])        # TOP_K najpopularniejszych
    coverage = top.sum() / len(df) * 100       # % pokrycia obserwacji
    
    # Podmieniamy rzadkie kategorie na 'Other'
    df[col] = df[col].where(df[col].isin(top.index), 'Other')
    
    # Info
    # print(f"{col}: TOP {TOP_K[col]} kategorii obejmuje {coverage:.2f}% wierszy "
    #       f"({top.sum()} z {len(df)})")
    

# Zmieniamy zmienne binarne na zero-jedynkowe i dodajemy do kolumn liczbowych
df[BIN_COLS] = df[BIN_COLS].replace(BINARY_MAP).astype(int)



# Badanie stacjonarności zmiennych i eliminacja niestacjonarności - pomijamy ponieważ badanie dotyczy danych przekrojowych

# LOGARYTMOWNIE - narazie pomijamy, gdyż może nie być potrzebne
# to_log = [c for c in NUM_COLS if abs(skew(df[c])) > 1]
# print("Logarytmuję:", to_log)

# for c in to_log:
#     df[c + '_log'] = np.log1p(df[c])





# from scipy.stats import skew, boxcox_normmax
# import statsmodels.api as sm
# import statsmodels.stats.api as sms

# # 1. Skew + Box‑Cox
# print("Skew price:", skew(df['Price_euros']))
# print("Box‑Cox λ:", boxcox_normmax(df['Price_euros'] + 1))

# # 2. Model bez log
# model_lin = smf.ols("Price_euros ~ ...", data=df).fit()
# res_lin = model_lin.resid
# print("BP p‑value (lin):", sms.het_breuschpagan(res_lin, model_lin.model.exog)[1])

# # 3. Model z log
# df['Price_euros_log'] = np.log(df['Price_euros'])
# model_log = smf.ols("Price_euros_log ~ ...", data=df).fit()
# res_log = model_log.resid
# print("BP p‑value (log):", sms.het_breuschpagan(res_log, model_log.model.exog)[1])


# Wartości odstające

# zwraca (dolna_granica, górna_granica) wg. reguły k*IQR.
def iqr_bounds(series, k=1.5):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr



# Zwraca DataFrame z liczbą i % outlierów w każdej kolumnie.
# Niczego nie zmienia w danych. Do sprawdzenia ile wartości odstających
def iqr_report(df: pd.DataFrame, cols, k=1.5):
    rows = []
    for c in cols:
        low, high = iqr_bounds(df[c], k)
        n_out = (~df[c].between(low, high)).sum()
        rows.append([c, n_out, 100 * n_out / len(df), low, high])

    rep = pd.DataFrame(rows,
                       columns=['kolumna', 'odstających', '%', 'dolna', 'górna'])
    rep = rep.sort_values('%', ascending=False)
    return rep


# Po przeanalizowaniu informacji odstających, uzjanemy na początku nie usuwać zadnych żadnych wartości odstających,
# gdyż dla większości nich np. jak dla `ScreenW` 80 % laptopów ma dokładnie 1920px wysokości. Każda inna wartość jest traktowana jako “odstająca”, a można też patrzeć na tą zmienną jako kategoryczną.
# To samo zjawisko można zauważyć dla kolumn `PrimaryStorage`, `Ram`, `SecondaryStorage`.
# Na usunięcie odstających wartości decydujemy się dla kolumn `Weight`, `Inches`.
# W dalszej części budowy modelu być może zdecydujemy się na usunięcię z innych pojedynczych zmiennych odstające wartości.

# Zwraca nowy DataFrame bez wierszy uznanych za odstające w podanych kolumnach.
def iqr_filter(df: pd.DataFrame, cols, k=1.5):
    mask = pd.Series(True, index=df.index)
    for c in cols:
        low, high = iqr_bounds(df[c], k)
        mask &= df[c].between(low, high)
    return df[mask].copy()

iqr_report(df, NUM_COLS)


before = len(df)
df_clean = iqr_filter(df, ['Weight', 'Inches'])

print(f"\nUsunięto {before - len(df_clean)} obserwacji "
      f"({100*(before - len(df_clean))/before:.2f} %).")


categorical_cols = CAT_TOPK_COLS + CAT_SMALL_COLS
# zmienne oznaczone w formule C(...) są autoamtycznie przetwarzane na zmienne zero-jedynkowe dla modelu ekoenometrycznego z pominięciem pierwszej kategorii jako bazowej.
rhs_terms = NUM_COLS + BIN_COLS + [f"C({c})" for c in categorical_cols]

formula = "Price_euros ~ " + " + ".join(rhs_terms)
print("\nFormuła OLS:\n", formula, "\n")

ols_model = smf.ols(formula=formula, data=df_clean).fit()

print(ols_model.summary())


# ZADANIE 3: METODA DOBORU ZMIENNYCH

# Przygotowanie danych dla metod doboru zmiennych
# Tworzymy DataFrame z wszystkimi zmiennymi objaśniającymi w postaci liczbowej

# Funkcja do kodowania zmiennych kategorycznych jako dummy variables
def create_dummy_vars(df, categorical_cols):
    df_dummies = df.copy()
    
    for col in categorical_cols:
        # Tworzymy zmienne dummy dla każdej kategorii (pomijamy pierwszą jako referencyjną)
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df_dummies = pd.concat([df_dummies, dummies], axis=1)
        df_dummies = df_dummies.drop(columns=[col])
    
    return df_dummies

# Przygotowanie danych
X_data = df_clean[NUM_COLS + BIN_COLS + categorical_cols].copy()
y_data = df_clean['Price_euros'].copy()

# Kodowanie zmiennych kategorycznych
X_encoded = create_dummy_vars(X_data, categorical_cols)
print(f"Liczba zmiennych objaśniających po kodowaniu: {X_encoded.shape[1]}")

# Import metod naprawczych
from corrective_methods import logarithmic_transformation, structural_break_correction, ramsey_reset_correction

if __name__ == "__main__":
    # Wywołanie model_creation.py do stworzenia i testowania modeli
    from model_creation import build_and_test_models
    
    # Przygotowanie danych dla modelu
    build_and_test_models(X_encoded, y_data, categorical_cols, NUM_COLS, BIN_COLS, df_clean)

