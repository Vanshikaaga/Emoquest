import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.ensemble import RandomForestClassifier
import os

def run_behavior_analysis():
    # Load data
    with open('reaction_logs.json', 'r') as file:
        log_data = json.load(file)

    df = pd.DataFrame(log_data)

    # Replace NaN values with 0 (you can change this to another value)
    df = df.fillna(0)

    # Convert timestamps
    df["spawn_time"] = pd.to_datetime(df["spawn_time"], unit='s')
    df["click_time"] = pd.to_datetime(df["click_time"], unit='s', errors='coerce')

    # Ensure that the necessary columns are numeric
    df["object_x"] = pd.to_numeric(df["object_x"], errors='coerce')
    df["object_y"] = pd.to_numeric(df["object_y"], errors='coerce')
    df["object_width"] = pd.to_numeric(df["object_width"], errors='coerce')
    df["object_height"] = pd.to_numeric(df["object_height"], errors='coerce')
    df["click_x"] = pd.to_numeric(df["click_x"], errors='coerce')
    df["click_y"] = pd.to_numeric(df["click_y"], errors='coerce')

    # Convert reaction time if missing
    if "reaction_time" not in df or df["reaction_time"].isnull().all():
        df["reaction_time"] = (df["click_time"] - df["spawn_time"]).dt.total_seconds()

    # Calculate object center and click distance
    df["object_center_x"] = df["object_x"] + df["object_width"] / 2
    df["object_center_y"] = df["object_y"] + df["object_height"] / 2

    # Ensure valid values before applying sqrt
    df["click_distance"] = np.sqrt(
        (df["click_x"] - df["object_center_x"]) ** 2 +
        (df["click_y"] - df["object_center_y"]) ** 2
    )

    # Handle missing or invalid click distances (optional)
    df["click_distance"] = df["click_distance"].fillna(0)

    # Compute other metrics
    df["click_inside_object"] = (
        (df["click_x"] >= df["object_x"]) & (df["click_x"] <= df["object_x"] + df["object_width"]) &
        (df["click_y"] >= df["object_y"]) & (df["click_y"] <= df["object_y"] + df["object_height"])
    )

    # Impulsivity analysis
    fast_clicks_under_150ms = len(df[df["reaction_time"] < 0.150])
    premature_clicks = len(df[(df["reaction_time"].isna()) & (df["clicked"] == False)])
    valid_clicks = df[df["clicked"] == True]
    impulsivity_index = fast_clicks_under_150ms / len(valid_clicks) if len(valid_clicks) > 0 else 0

    # Misclicks analysis
    df["misclicks"] = df.apply(lambda row: row["clicked"] and (
        row["target"] == "rat" or not row["click_inside_object"]
    ), axis=1)

    misclicks_per_minute = df["misclicks"].sum() / (
        (df["spawn_time"].max() - df["spawn_time"].min()).total_seconds() / 60
    )

    # Reaction time curve
    reaction_time_curve = df.groupby(df["spawn_time"].dt.minute)["reaction_time"].mean()

    # Startled responses
    startled_threshold = 0.5
    startled_responses = df[(df["reaction_time"] > startled_threshold) & (df["clicked"] == True)]

    # Misclick streaks
    df["misclick_streak_group"] = (df["misclicks"] != df["misclicks"].shift()).cumsum()
    misclick_streaks = df[df["misclicks"] == True].groupby("misclick_streak_group").size().max()
    if pd.isna(misclick_streaks):
        misclick_streaks = 0

    # Summary stats
    mean_click_distance = df["click_distance"].mean()
    click_accuracy_ratio = df["click_inside_object"].sum() / len(valid_clicks) if len(valid_clicks) > 0 else 0

    # Feature vector for behavior prediction
    features = pd.DataFrame({
        "mean_reaction_time": [df["reaction_time"].mean()],
        "std_reaction_time": [df["reaction_time"].std()],
        "fast_clicks_under_150ms": [fast_clicks_under_150ms],
        "premature_clicks": [premature_clicks],
        "impulsivity_index": [impulsivity_index],
        "misclicks_per_minute": [misclicks_per_minute],
        "max_false_positive_streak": [misclick_streaks],
        "mean_click_distance": [mean_click_distance],
        "click_accuracy_ratio": [click_accuracy_ratio],
        "startled_responses_count": [len(startled_responses)]
    })

    # Predict behavior using RandomForest (simplified)
    clf = RandomForestClassifier()
    X_train = features.copy()
    y_train = [1]  # Assume 'Focused' for training
    clf.fit(X_train, y_train)

    predicted_behavior = clf.predict(features)[0]
    behavior_label = {
        0: "Impulsive",
        1: "Focused",
        2: "Startled"
    }.get(predicted_behavior, "Unknown")

    # Report
    report = {
        "mean_reaction_time": features["mean_reaction_time"].iloc[0],
        "std_reaction_time": features["std_reaction_time"].iloc[0],
        "fast_clicks_under_150ms": fast_clicks_under_150ms,
        "premature_clicks": premature_clicks,
        "impulsivity_index": impulsivity_index,
        "misclicks_per_minute": misclicks_per_minute,
        "click_accuracy_ratio": click_accuracy_ratio,
        "mean_click_distance": mean_click_distance,
        "reaction_time_curve": reaction_time_curve.tolist(),
        "startled_responses_count": len(startled_responses),
        "max_false_positive_streak": misclick_streaks,
        "predicted_behavior": behavior_label
    }

    print("Detailed Player Behavior Report\n")
    for key, value in report.items():
        print(f"{key}: {value}")

    # Plot reaction curve
    plt.figure(figsize=(12, 5))
    sns.lineplot(x=reaction_time_curve.index, y=reaction_time_curve.values)
    plt.title("Average Reaction Time Over Minutes")
    plt.xlabel("Minute")
    plt.ylabel("Avg Reaction Time (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Append report to JSON file
    if os.path.isfile('report.json'):
        with open('report.json', 'r') as file:
            existing_reports = json.load(file)
            # Ensure existing_reports is a list, not a dictionary
            if not isinstance(existing_reports, list):
                existing_reports = [existing_reports]  # Convert to list if necessary
    else:
        existing_reports = []

    # Now that we have ensured existing_reports is a list, we can append the report
    existing_reports.append(report)

    # Save the updated list back to report.json
    with open('report.json', 'w') as file:
        json.dump(existing_reports, file, indent=4)

    print("Report appended to report.json")
