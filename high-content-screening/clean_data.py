import os
import shutil
import random
from collections import defaultdict

data_dir = "data"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# files_by_sample pairs up DAPI and TRANS images using their common prefix
# e.g., "G1_10x_A3_F1_T0" -> ["G1_10x_A3_F1_T0_DAPI.jpg", "G1_10x_A3_F1_T0_TRANS.jpg"]
files_by_sample = defaultdict(list)

# samples_by_combo groups the sample prefixes by (group, mag, treatment)
samples_by_combo = defaultdict(list)

for file in os.listdir(data_dir):
    # Skip the train/test directories if they are already inside "data"
    if not os.path.isfile(os.path.join(data_dir, file)):
        continue
        
    # Clean up filename as requested
    new_name = file.replace("-", "_").replace("0.5", "5")
    
    old_path = os.path.join(data_dir, file)
    new_path = os.path.join(data_dir, new_name)
    
    if old_path != new_path:
        os.rename(old_path, new_path)
    
    # Parse the filename. Example: G1_10x_A3_F1_T0_DAPI.jpg
    name_without_ext = new_name.split(".")[0]
    parts = name_without_ext.split("_")
    
    # Ensure it's one of your target image files (needs at least 6 parts based on the screenshot)
    if len(parts) >= 6:
        group = parts[0]
        mag = parts[1]
        treatment = parts[4] 
        
        # The base sample identifier (everything except the image type like DAPI/TRANS)
        sample_id = "_".join(parts[:-1])
        
        files_by_sample[sample_id].append(new_name)
        
        # Add the sample_id to the combination dictionary (ensure no duplicates)
        if sample_id not in samples_by_combo[(group, mag, treatment)]:
            samples_by_combo[(group, mag, treatment)].append(sample_id)

# Set random seed for reproducibility
random.seed(1)
split_ratio = 0.8 # 80% to train, 20% to test

# Split and move files
for combo, sample_ids in samples_by_combo.items():
    # Shuffle the sample IDs to randomize the split for this specific combination
    random.shuffle(sample_ids)
    
    # Calculate where to split the list
    split_idx = int(len(sample_ids) * split_ratio)
    
    train_samples = sample_ids[:split_idx]
    test_samples = sample_ids[split_idx:]
    
    # Move train files
    for sample_id in train_samples:
        for file_name in files_by_sample[sample_id]:
            src = os.path.join(data_dir, file_name)
            dst = os.path.join(train_dir, file_name)
            shutil.move(src, dst)
            
    # Move test files
    for sample_id in test_samples:
        for file_name in files_by_sample[sample_id]:
            src = os.path.join(data_dir, file_name)
            dst = os.path.join(test_dir, file_name)
            shutil.move(src, dst)
            
print("Data stratification and splitting complete!")