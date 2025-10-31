import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('test/cold test/data/LOG00126.csv', encoding='utf-8-sig')

df['time_s'] = df['time'] / 1000

# Plot Temperature vs Time
plt.figure(1)
plt.plot(df["time_s"], df["temp"])
plt.title("Time vs Temperature")
plt.xlabel("Time")
plt.ylabel("Temperature")

# Plot Predicted Layer vs Time
plt.figure(2)
plt.plot(df["time_s"], df["predicted_layer"])
plt.title("Time vs Predicted Layer Classification")
plt.xlabel("Time (ms)")
plt.ylabel("Predicted Layer")

plt.show()
