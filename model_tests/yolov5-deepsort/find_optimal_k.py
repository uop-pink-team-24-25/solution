import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import glob

def calculate_wcss(data, k_max):
    wcss = []
    reshaped_data = data.reshape((-1, 3))

    if reshaped_data.dtype != np.float32:
         reshaped_data = np.array(reshaped_data, dtype=np.float32)

    effective_k_max = min(k_max, len(np.unique(reshaped_data, axis=0)))
    if effective_k_max < 1:
         return None

    print(f"  Calculating WCSS for k=1 to {effective_k_max}...")
    for k in range(1, effective_k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', verbose=0)
        kmeans.fit(reshaped_data)
        wcss.append(kmeans.inertia_)
        print(f"    k={k}, WCSS={kmeans.inertia_:.2f}")
    return wcss

sample_dir = "./kmeans_samples"
output_plot_file = "kmeans_elbow_plot.png"
max_k_to_test = 10

image_files = glob.glob(os.path.join(sample_dir, "*.png"))

if not image_files:
    print(f"Error: No sample images found in {sample_dir}")
    exit()

print(f"Found {len(image_files)} sample images.")

all_wcss_values = []
valid_image_count = 0

for i, img_file in enumerate(image_files):
    print(f"Processing image {i+1}/{len(image_files)}: {img_file}")
    img = cv2.imread(img_file)
    if img is None:
        print(f"  Warning: Could not read image {img_file}")
        continue

    wcss_single = calculate_wcss(img, max_k_to_test)

    if wcss_single:
         if len(wcss_single) < max_k_to_test:
             wcss_single.extend([np.nan] * (max_k_to_test - len(wcss_single)))
         all_wcss_values.append(wcss_single)
         valid_image_count += 1
    else:
        print(f"  Skipping image {img_file} due to insufficient unique colors.")

if valid_image_count == 0:
    print("Error: No valid images processed to generate elbow plot.")
    exit()

all_wcss_array = np.array(all_wcss_values)
average_wcss = np.nanmean(all_wcss_array, axis=0)

k_values = range(1, max_k_to_test + 1)

plt.figure(figsize=(10, 6))
plt.plot(k_values, average_wcss, marker='o', linestyle='-')
plt.title(f'Elbow Method for Optimal K (Average over {valid_image_count} images)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Average Within-Cluster Sum of Squares (WCSS/Inertia)')
plt.xticks(k_values)
plt.grid(True)
plt.savefig(output_plot_file)
print(f"\nElbow plot saved to {output_plot_file}")
plt.show()

try:
    pct_drop = [(average_wcss[i] - average_wcss[i+1]) / average_wcss[i] * 100 for i in range(len(average_wcss)-1)]
    print("\nPercentage drop in WCSS:")
    for k_idx, drop in enumerate(pct_drop):
         print(f"  K={k_idx+1} -> K={k_idx+2}: {drop:.2f}%")

    elbow_k = 2
    for k_idx, drop in enumerate(pct_drop):
        if k_idx > 0 and drop < 15:
            elbow_k = k_idx + 1
            break
    print(f"\nSuggested Elbow Point (heuristic): K = {elbow_k}")
except Exception as e:
    print(f"\nCould not automatically suggest elbow: {e}")