import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

def plot_correlation_heatmap(X_data: pd.DataFrame, target_variable: str = None):
    """
    Tworzy mapę cieplną korelacji dla zmiennych numerycznych.
    Pomija kolumny stałe i nieskorelowalne (np. same 0/1 bez wariancji).
    """
    # Usuń kolumnę 'const' jeśli występuje
    X = X_data.drop(columns=['const'], errors='ignore').copy()

    # Upewnij się, że wszystko jest typu float (szczególnie dummy variables)
    X = X.apply(pd.to_numeric, errors='coerce')

    # Usuń kolumny, które mają zerowe odchylenie standardowe (same 1 albo same 0)
    X = X.loc[:, X.std() != 0]

    # Oblicz korelacje tylko dla zmiennych liczbowych
    numeric_cols = X.select_dtypes(include='number').columns
    corr = X[numeric_cols].corr()

    # Rysowanie wykresu
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, vmin=-1, vmax=1)

    ax.set_xticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=90, ha="center")
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_yticklabels(numeric_cols)

    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha='center', va='center', color='black')

    if target_variable in numeric_cols:
        row_index = list(numeric_cols).index(target_variable)
        rect = patches.Rectangle(
            (-0.5, row_index - 0.5),
            len(numeric_cols),
            1,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Współczynnik korelacji", rotation=90)
    plt.title("Macierz korelacji – heatmap")
    plt.tight_layout()
    plt.show()
