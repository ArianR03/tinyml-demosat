import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('test/bench test/data/LOG00127.csv', encoding='utf-8-sig')

df['time_s'] = df['time'] / 1000

# Plot Temperature vs Time
plt.figure(1)
plt.plot(df["time_s"], df["x_acc"], label="X acceleration")
plt.plot(df["time_s"], df["y_acc"], label="Y acceleration")
plt.plot(df["time_s"], df["z_acc"], label="Z acceleration")
plt.title("Whip Test")
plt.xlabel("Time")
plt.ylabel("Acceleration")
plt.legend()
plt.show()

