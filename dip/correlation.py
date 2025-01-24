import numpy as np
import matplotlib.pyplot as plt


def plot_and_correlate(signal1, signal2, segment_length):
    # Plotting the signals
    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    plt.plot(signal1, label="Signal 1")
    plt.plot(signal2, label="Signal 2")
    plt.title("Signals")
    plt.legend()

    # Computing correlation between segments
    correlation_values = []
    for i in range(len(signal1) - segment_length + 1):
        segment1 = signal1[i : i + segment_length]
        segment2 = signal2[i : i + segment_length]
        correlation = np.correlate(segment1, segment2)
        correlation_values.append(correlation)

    # Plotting the correlation values
    plt.subplot(2, 1, 2)
    plt.plot(correlation_values, label="Correlation")
    plt.title("Correlation between segments")
    plt.legend()

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Generate example signals
    fs = 100  # Sampling frequency
    segment_length = 10
    t = np.linspace(0, 1, fs, endpoint=False)
    signal1 = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave
    signal2 = np.sin(2 * np.pi * 5 * t + 3 * np.pi / 2)  # Shifted sine wave

    # Call the function to plot and compute correlation
    plot_and_correlate(signal1, signal2, segment_length)
