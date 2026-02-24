import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Define patterns and setup variables
file_patterns = {
    'OpenCV Contours': 'colony_data_opencv{}_black.csv',
    'SAM Model': 'colony_measurements{}_black.csv',
    'Ground Truth': 'red_shape_areas{}.csv'
}

images = [1, 2, 3, 4]
methods = list(file_patterns.keys())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green

# Dictionary to hold the detection counts for each method
counts_data = {method: [] for method in methods}

# 2. Extract the row counts (detections) from each CSV
for method in methods:
    for img_num in images:
        filename = file_patterns[method].format(img_num)
        
        if os.path.exists(filename):
            try:
                # Read the CSV and count the number of rows
                df = pd.read_csv(filename)
                counts_data[method].append(len(df))
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                counts_data[method].append(0)
        else:
            # If the file is missing (e.g., Image 2 for Red Shape), record 0
            counts_data[method].append(0)

# 3. Create the Bar Chart
x = np.arange(len(images))  # Base X locations for the groups
bar_width = 0.25            # Width of the individual bars

fig, ax = plt.subplots(figsize=(10, 6))

# Offsets to place bars side-by-side: left, center, right
offsets = [-bar_width, 0, bar_width]

# 4. Plot each method's bars
for i, method in enumerate(methods):
    # Create the bars
    rects = ax.bar(x + offsets[i], counts_data[method], bar_width, 
                   label=method, color=colors[i], alpha=0.85)
    
    # Automatically add the exact count text on top of each bar
    ax.bar_label(rects, padding=3, fontsize=10)

# 5. Formatting and Labels
ax.set_ylabel('Number of Colonies', fontsize=12)
ax.set_xlabel('Source Image', fontsize=12)
ax.set_title('Total detections for White Background', fontsize=14)

# Set the x-ticks to the center of the groups
ax.set_xticks(x)
ax.set_xticklabels([f'Image {i}' for i in images], fontsize=11)

# Add legend and grid
ax.legend(title="Method", loc='upper right')
# ax.grid(axis='y', linestyle='--', alpha=0.6)

# Ensure everything fits without getting cut off
fig.tight_layout()

# 6. Show the plot
plt.show()