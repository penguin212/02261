import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Define the file structure patterns and methods
# The {} will be replaced by the image number (1, 2, 3, 4)
file_patterns = {
    'OpenCV Contours': 'colony_data_opencv{}.csv',
    'SAM Model': 'colony_measurements{}.csv',
    'Ground Truth': 'red_shape_areas{}.csv'
}

images = [1, 2, 3, 4]
methods = list(file_patterns.keys())

# Colors for our 3 methods
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green

# Variables to hold data for matplotlib
plot_data = []
positions = []
box_colors = []

# Spacing configurations for the grouped boxplot
group_spacing = 1.0
bar_width = 0.2
# Offsets to place the 3 boxes side-by-side for each image
offsets = [-bar_width, 0, bar_width]

# 2. Read the CSVs and organize the data
for img_idx, img_num in enumerate(images):
    base_pos = (img_idx + 1) * group_spacing
    
    for method_idx, method in enumerate(methods):
        filename = file_patterns[method].format(img_num)
        
        # Calculate where this specific box will go on the X-axis
        pos = base_pos + offsets[method_idx]
        
        # Check if file exists (handles the missing red_shape_areas2.csv gracefully)
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                # Ensure the column exists and drop any NaNs
                if 'Area_Pixels' in df.columns:
                    areas = df['Area_Pixels'].dropna().values
                    plot_data.append(areas)
                else:
                    plot_data.append([]) # Empty list if column is missing
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                plot_data.append([])
        else:
            # File missing, add empty list to maintain positioning
            plot_data.append([])
            
        positions.append(pos)
        box_colors.append(colors[method_idx])

# 3. Create the Plot
plt.figure(figsize=(12, 7))

# Create the boxplot
bplot = plt.boxplot(plot_data, 
                    positions=positions, 
                    widths=bar_width * 0.8, 
                    patch_artist=True,  # Allows us to fill with color
                    showfliers=True)    # Shows outliers

# 4. Color the boxes based on the method
for patch, color in zip(bplot['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    
# Color the medians black for visibility
for median in bplot['medians']:
    median.set(color='black', linewidth=1.5)

# 5. Formatting the Chart
# Set X-ticks to be exactly in the middle of the groups
plt.xticks([i * group_spacing for i in range(1, len(images) + 1)], 
           [f'Image {i}' for i in images], 
           fontsize=12)

# Use a log scale because areas can vary massively (e.g. 10 pixels vs 50,000 pixels)
plt.yscale('log')
plt.ylabel('Area (Pixels) - Log Scale', fontsize=12)
plt.xlabel('Source Image', fontsize=12)
plt.title('Comparison of Segmentation Methods by Area Distribution', fontsize=14)

# Add a custom grid behind the boxes for readability
plt.grid(axis='y', linestyle='--', alpha=0.6)

# 6. Create a custom legend
# We create "dummy" patches with the correct colors to map to our methods
legend_patches = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7) for i in range(len(methods))]
plt.legend(legend_patches, methods, title="Method")

# Add some padding to the x-axis limits to make it look nicer
plt.xlim(0.5, len(images) + 0.5)

plt.tight_layout()
plt.show()