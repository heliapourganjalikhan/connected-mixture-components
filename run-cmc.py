import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from src.cmc_model import ConnectedMixtureComponents

#  Load ETTh1 dataset
df = pd.read_csv("data/ETTh1.csv")
values = df["OT"].values.reshape(-1, 1)

#  Scale data
scaler = MinMaxScaler()
X = scaler.fit_transform(values)

#  Run CMC model
cmc = ConnectedMixtureComponents(n_components=4, window_size=168, step_size=24, epsilon=0.5)
features = cmc.fit_transform(X)
stats = cmc.get_component_stats()

#  Plot time-series
plt.figure(figsize=(10,4))
plt.plot(df["OT"], color="blue", label="Oil Temperature")
plt.xlabel("Time Step")
plt.ylabel("Temperature")
plt.title("Oil Temperature Over Time")
plt.legend()
plt.tight_layout()
plt.savefig("visualizations/oil_temperature_over_time.png")

#  Plot component statistics
cmc.plot_component_stats()
