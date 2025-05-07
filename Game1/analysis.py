import pandas as pd
import numpy as np
import os
import json

def run_analysis():
    # Load your CSV or dataframe
    df = pd.read_csv('game_data.csv')

    # --- Step 4: Compute Derived Metrics ---
    df['Time'] = df['Time'].astype(float)

    # avg_distance
    window_size = int(3 / (df['Time'].diff().median()))
    df['avg_distance'] = df['Distance_to_Nearest_Enemy'].rolling(window=window_size, min_periods=1).mean()

    # time_in_safe_area
    df['time_in_safe_area'] = df['Time_in_Safe_Areas'].cumsum()

    # reaction_latency
    df['distance_delta'] = df['Distance_to_Nearest_Enemy'].diff()
    df['reaction_trigger'] = (df['distance_delta'] > 0) & (df['distance_delta'].shift(1) < 0)
    df['reaction_latency'] = np.where(df['reaction_trigger'], df['Time'].diff().fillna(0), 0)
    df['reaction_latency'] = df['reaction_latency'].replace(0, np.nan).fillna(method='ffill').fillna(0)

    # speed_mean
    df['speed_mean'] = df['Player_Speed'].rolling(window=window_size, min_periods=1).mean()

    # input_rate
    df['Input_Frequency'] = df['Input_Frequency'].astype(float)
    df['input_rate'] = df['Input_Frequency'].rolling(window=window_size, min_periods=1).mean()

    # evade_count
    evade_threshold = df['Distance_to_Nearest_Enemy'].std()
    df['evade'] = df['distance_delta'] > evade_threshold
    df['evade_count'] = df['evade'].cumsum()

    # Normalization
    df['normalized_distance'] = df['Distance_to_Nearest_Enemy'] / 1000
    df['normalized_speed'] = df['Player_Speed'] / df['Player_Speed'].max()
    df['normalized_input_rate'] = df['input_rate'] / df['input_rate'].max()

    df.drop(columns=['distance_delta', 'reaction_trigger', 'evade'], inplace=True)
    df.to_csv('engineered_features.csv', index=False)

    # Summaries
    print("=== Feature Summary Statistics ===")
    print(df[['avg_distance', 'time_in_safe_area', 'reaction_latency',
              'speed_mean', 'input_rate', 'evade_count',
              'normalized_distance', 'normalized_speed', 'normalized_input_rate']].describe())

    print("\n=== Feature Correlations ===")
    print(df[['avg_distance', 'reaction_latency', 'speed_mean', 'input_rate']].corr())

    print("\n=== Sample Feature Trends (first 10 rows) ===")
    print(df[['Time', 'avg_distance', 'reaction_latency', 'speed_mean', 'input_rate', 'evade_count']].head(10))

    evades = df['evade_count'].iloc[-1]
    print(f"\n Total Evade Events Detected: {evades}")

    avg_latency = df['reaction_latency'][df['reaction_latency'] > 0].mean()
    print(f" Average Reaction Latency: {avg_latency:.3f} seconds")

    safe_time = df['time_in_safe_area'].iloc[-1]
    total_time = df['Time'].iloc[-1]
    safe_ratio = safe_time / total_time * 100 if total_time > 0 else 0

    # Final values for report
    report = {
        'total_evades': evades,
        'average_reaction_latency': avg_latency,
        'safe_area_time': safe_time,
        'total_time': total_time,
        'safe_area_percentage': safe_ratio
    }

    report_filename = 'report.json'
    def replace_nan_with_default(report, default_value=0):
        for key, value in report.items():
            if pd.isna(value):  # Check if the value is NaN
                report[key] = default_value
        return report
    # Function to convert non-serializable objects
    report = replace_nan_with_default(report, default_value=0)
    def convert(o):
        if isinstance(o, (np.integer, np.int32, np.int64)):
            return int(o)
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        return str(o)

    # Write new report (overwrite file)
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=4, default=convert)

    print(f"\nReport saved to {report_filename}")

