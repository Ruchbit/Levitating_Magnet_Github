import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def smooth_data(data, window_size):
    """
    Smooths the data using a moving average.

    Parameters:
        data (pd.DataFrame): The input data.
        window_size (int): The window size for the moving average.

    Returns:
        pd.DataFrame: The smoothed data.
    """
    return data.rolling(window=window_size, min_periods=1).mean()

def plot_sensor_data(file_path, start_index, end_index, window_size=5, sample_rate_ms=5):
    """
    Plots sensor data from a CSV file within a specified range.

    Parameters:
        file_path (str): The path to the CSV file.
        start_index (int): The starting index of the range to plot.
        end_index (int): The ending index of the range to plot.
        window_size (int): The window size for the moving average.
        sample_rate_ms (int): The sample rate in milliseconds.
    """
    # Read CSV file
    data = pd.read_csv(file_path, header=None, names=['X', 'Y', 'Z'], index_col=False)

    # Replace 0 values with the mean of the respective column
    data.replace(0, np.nan, inplace=True)
    data.fillna(data.mean(), inplace=True)

    # Smooth the data
    smoothed_data = smooth_data(data, window_size)

    # Plotting
    plt.figure(figsize=(10, 6))
    x = np.arange(start_index, end_index) * (sample_rate_ms / 1000.0)  # Time in seconds
    plt.plot(smoothed_data['X'][start_index:end_index], label='X', linewidth=2)
    plt.plot(smoothed_data['Y'][start_index:end_index], label='Y', linewidth=2)
    plt.plot(smoothed_data['Z'][start_index:end_index], label='Z', linewidth=2)

    # Chart details
    plt.title('Sensor Data (X, Y, Z)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Sensor Value')
    plt.legend()
    plt.grid(True)

    # Customize y-axis ticks and limits
    plt.ylim([min(smoothed_data[['X', 'Y', 'Z']].min()) - 10, max(smoothed_data[['X', 'Y', 'Z']].max()) + 10])
    plt.yticks(np.arange(min(smoothed_data[['X', 'Y', 'Z']].min()) - 10, max(smoothed_data[['X', 'Y', 'Z']].max()) + 10, 10))

    # Show plot
    plt.show()