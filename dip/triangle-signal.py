import numpy as np
import matplotlib.pyplot as plt

fs = 100.0
t_period = 1.0
t_duration = 3 * t_period
amplitude = 1.0
offset = 0.0

t = np.arange(0, t_duration, 1 / fs)

signal = amplitude * np.abs(2 * (t % t_period - t_period / 2)) + offset
# print (signal)
# Plotting
plt.figure(figsize=(12, 6))
plt.plot(t, signal, label="Triangle Wave")
plt.title("Triangle Wave - 3 Periods")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()
