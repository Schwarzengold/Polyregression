import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

def main():
    df = pd.read_csv('fuel_consumption_vs_speed.csv')
    X = df[['speed_kmh']].values
    y = df['fuel_consumption_l_per_100km'].values

    print("Порівняння моделей поліноміальної регресії за ступенем:")
    best_mse = float('inf')
    best_degree = 0
    for deg in range(1, 6):
        model = make_pipeline(PolynomialFeatures(degree=deg), LinearRegression())
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        print(f"  Ступінь {deg}: MSE = {mse:.4f}, MAE = {mae:.4f}")
        if mse < best_mse:
            best_mse = mse
            best_degree = deg

    print(f"\nНайкращий ступінь полінома (min MSE): {best_degree}")

    best_model = make_pipeline(PolynomialFeatures(degree=best_degree), LinearRegression())
    best_model.fit(X, y)
    test_speeds = np.array([35, 95, 140]).reshape(-1, 1)
    preds = best_model.predict(test_speeds)

    print("\nПрогноз витрати палива для заданих швидкостей:")
    for speed, fuel in zip([35, 95, 140], preds):
        print(f"  при {speed} км/ч: {fuel:.2f} л/100 км")

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color='tab:blue', label='Вихідні дані', s=50, edgecolor='k')
    X_plot = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_plot = best_model.predict(X_plot)
    plt.plot(
        X_plot, y_plot,
        color='tab:orange', linewidth=2,
        label=f'Поліном {best_degree}-го ступеня'
    )
    plt.xlabel('Швидкість, км/ч')
    plt.ylabel('Витрата палива, л/100 км')
    plt.title('Поліноміальна регресія: витрата палива vs швидкість')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
