import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import requests
import folium
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import time

FOURSQUARE_API_KEY = "fsq3JHxmemIymJphWwit01HbvNvBbgXkms+v3wPai/ejzsA="
RADIUS_METERS = 20
MIN_SAMPLES = 5
SLEEP_BETWEEN_CALLS = 1

headers = {
    "Accept": "application/json",
    "Authorization": FOURSQUARE_API_KEY
}


def get_location_label(lat, lon):
    url = f"https://api.foursquare.com/v3/places/search?ll={lat},{lon}&radius=250&limit=1"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if data["results"]:
            result = data["results"][0]
            categories = result.get("categories", [])
            if categories:
                return categories[0]["name"]
            return result.get("name", "Unknown")
        else:
            return "No results"
    else:
        print(f"API Error {response.status_code}: {response.text}")
        return "API Error"


# Use of AI (ChatGPT) to incorporate DBSCAN in identifying significant locations that were stayed for a long time
def process_user(file_path, user_label):
    print(f"\n Processing {user_label} from {file_path}")
    df = pd.read_csv(file_path, index_col=False)
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df = df[['datetime', 'latitude', 'longitude']].dropna()

    coordinatess = df[['latitude', 'longitude']].to_numpy()
    kms_per_radian = 6371.0088
    epsilon = RADIUS_METERS / 1000.0 / kms_per_radian

    db = DBSCAN(eps=epsilon, min_samples=MIN_SAMPLES, algorithm='ball_tree', metric='haversine').fit(
        np.radians(coordinatess))
    df['cluster'] = db.labels_

    clusters = df[df['cluster'] != -1].groupby('cluster')[['latitude', 'longitude']].mean().reset_index()

    labels = []
    for i, row in clusters.iterrows():
        print(f"Querying {user_label} location {i + 1}/{len(clusters)} at ({row['latitude']}, {row['longitude']})...")
        label = get_location_label(row['latitude'], row['longitude'])
        print(f" → {user_label} Label: {label}")
        labels.append(label)
        time.sleep(SLEEP_BETWEEN_CALLS)

    clusters['label'] = labels

    return df, clusters, Counter(labels)


df_a, clusters_a, counts_a = process_user("gps_u07.csv", "User A")
df_b, clusters_b, counts_b = process_user("gps_u08.csv", "User B")

# Use of AI (ChatGPT) to incorporate Folium in creating an interactive map
combined_latitude = pd.concat([df_a['latitude'], df_b['latitude']]).mean()
combined_longitude = pd.concat([df_a['longitude'], df_b['longitude']]).mean()
m = folium.Map(location=[combined_latitude, combined_longitude], zoom_start=15)

for _, row in clusters_a.iterrows():
    folium.Marker(
        [row['latitude'], row['longitude']],
        popup=f"User A: {row['label']}",
        icon=folium.Icon(color='blue')
    ).add_to(m)

for _, row in clusters_b.iterrows():
    folium.Marker(
        [row['latitude'], row['longitude']],
        popup=f"User B: {row['label']}",
        icon=folium.Icon(color='green')
    ).add_to(m)

m.save("significant_locations_map.html")
print("Map saved as 'significant_locations_map.html' with both users’ locations")

all_labels = list(set(counts_a.keys()).union(set(counts_b.keys())))
df_compare = pd.DataFrame({
    "Location Type": all_labels,
    "User A": [counts_a.get(label, 0) for label in all_labels],
    "User B": [counts_b.get(label, 0) for label in all_labels],
})

x = np.arange(len(df_compare["Location Type"]))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width / 2, df_compare["User A"], width=width, label="User A")
plt.bar(x + width / 2, df_compare["User B"], width=width, label="User B")
plt.xticks(x, df_compare["Location Type"], rotation=45, ha='right')
plt.xlabel("Location Type")
plt.ylabel("Frequency")
plt.title("Comparison of Significant Location Types: User A vs. User B")
plt.legend()
plt.tight_layout()
plt.show()
