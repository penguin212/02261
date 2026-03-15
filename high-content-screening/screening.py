import os
import pickle
from img_metrics import count_255s, pixel_histogram
from sklearn.ensemble import RandomForestClassifier
from cell_metrics import avg_cell_area
from tqdm import tqdm
import numpy as np

# --- Conceptual Imports ---
# Replace these with your actual imports from your other files
# e.g., from my_features import get_treatment_features, etc.

def extract_features(img):
    over_exp = count_255s(img)
    hist = pixel_histogram(img)
    avg_area, count = avg_cell_area(img) 
    return [over_exp, avg_area, count] + list(hist)

def extract_short_features(img):
    over_exp = count_255s(img)
    hist = pixel_histogram(img)
    return [over_exp] + list(hist)

blacklist = ["G6_20x_B6_F4_T0_TRANS.jpg", "G6_20x_F5_F1_T4_TRANSjpg.jpg"]

cull_rate = .5


# --- Directories ---
MODELS_DIR = "models"
TRAIN_DIR = os.path.join("data", "train")

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

def getDAPI(filename : str):
    return filename.replace("TRANS", "DAPI")

# Helper function to parse labels from the filename format we set up earlier
def _parse_labels(filename):
    """Extracts group, mag, and treatment integers from filenames like G1_10x_A3_F1_T0_DAPI.jpg"""
    parts = filename.split(".")[0].split("_")
    group_num = int(parts[0][1:])     # 'G1' -> 1
    mag_num = int(parts[1][:-1])      # '10x' -> 10
    treatment_num = int(parts[4][1:]) # 'T0' -> 0
    return group_num, mag_num, treatment_num


### --- TRAINING FUNCTIONS --- ###

def train_group():
    print("Training group model...")
    X, y = [], []
    for file in tqdm(os.listdir(TRAIN_DIR)):
        if file in blacklist:
            continue
        if("TRANS" not in file):
            continue
        group_val, _, _ = _parse_labels(file)
        img_path = os.path.join(TRAIN_DIR, file)
        other_img_path = getDAPI(img_path)
        
        X.append(extract_short_features(img_path) + extract_short_features(other_img_path))
        y.append(group_val)
        
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)
    
    with open(os.path.join(MODELS_DIR, 'group_model.pkl'), 'wb') as f:
        pickle.dump(clf, f)
    print("Group model saved.")

def train_mag():
    print("Training mag model...")
    X, y = [], []
    for file in tqdm(os.listdir(TRAIN_DIR)):
        if file in blacklist:
            continue
        if("TRANS" not in file):
            continue
        _, mag_val, _ = _parse_labels(file)
        img_path = os.path.join(TRAIN_DIR, file)
        other_img_path = getDAPI(img_path)
        
        X.append(extract_short_features(img_path) + extract_short_features(other_img_path))
        y.append(mag_val)
        
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)
    
    with open(os.path.join(MODELS_DIR, 'mag_model.pkl'), 'wb') as f:
        pickle.dump(clf, f)
    print("Mag model saved.")

def train_treatment():
    print("Training treatment model...")
    X, y = [], []
    for file in tqdm(os.listdir(TRAIN_DIR)):
        if file in blacklist:
            continue
        if("TRANS" not in file):
            continue
        if(np.random.rand() < cull_rate):
            continue
        _, _, treatment_val = _parse_labels(file)
        img_path = os.path.join(TRAIN_DIR, file)
        other_img_path = getDAPI(img_path)
        
        X.append(extract_features(img_path) + extract_features(other_img_path))
        y.append(treatment_val)
        
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)
    
    with open(os.path.join(MODELS_DIR, 'treatment_model.pkl'), 'wb') as f:
        pickle.dump(clf, f)
    print("Treatment model saved.")


### --- PREDICTION FUNCTIONS --- ###



def predict_group(img_path):
    model_path = os.path.join(MODELS_DIR, 'group_model.pkl')
    dapi_path = getDAPI(img_path)
    
    if not os.path.exists(model_path):
        train_group()
        
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
        
    features = extract_short_features(img_path) + extract_short_features(dapi_path)
    prediction = clf.predict([features])
    return prediction

def predict_mag(img_path):
    model_path = os.path.join(MODELS_DIR, 'mag_model.pkl')
    dapi_path = getDAPI(img_path)
    
    if not os.path.exists(model_path):
        train_mag()
        
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
        
    features = extract_short_features(img_path) + extract_short_features(dapi_path)
    prediction = clf.predict([features])
    return prediction

def predict_treatment(img_path):
    model_path = os.path.join(MODELS_DIR, 'treatment_model.pkl')
    dapi_path = getDAPI(img_path)
    
    if not os.path.exists(model_path):
        train_treatment()
        
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
        
    features = extract_features(img_path) + extract_features(dapi_path)
    prediction = clf.predict([features])
    return prediction

def get_group_error_rate(test_dir="data/test"):
    # Load model once to save time
    model_path = os.path.join(MODELS_DIR, 'group_model.pkl')
    if not os.path.exists(model_path):
        train_group()
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
        
    errors, total = 0, 0
    error_dict = dict()
    
    for file in tqdm(os.listdir(test_dir)):
        # Skip blacklisted files and DAPI images (since we process them in pairs)
        if file in blacklist or "TRANS" not in file:
            continue
            
        true_group, _, _ = _parse_labels(file)
        img_path = os.path.join(test_dir, file)
        
        # Combine features exactly as done in training
        features = extract_short_features(img_path) + extract_short_features(getDAPI(img_path))
        
        # Predict and check against ground truth
        predicted_group = clf.predict([features])[0]
        if true_group not in error_dict.keys():
            error_dict[true_group] = (0,0)
        
        curr_right, curr_total = error_dict[true_group]
        if predicted_group != true_group:
            errors += 1
        else:
            curr_right += 1
        curr_total += 1
        error_dict[true_group] = (curr_right, curr_total)
        total += 1
    
    print(error_dict)
    return errors / total if total > 0 else 0.0

def get_mag_error_rate(test_dir="data/test"):
    # Load model once to save time
    model_path = os.path.join(MODELS_DIR, 'mag_model.pkl')
    print(model_path)
    if not os.path.exists(model_path):
        train_mag()
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
        
    errors, total = 0, 0
    error_dict = dict()
    
    for file in tqdm(os.listdir(test_dir)):
        # Skip blacklisted files and DAPI images (since we process them in pairs)
        if file in blacklist or "TRANS" not in file:
            continue
            
        _, true_mag, _ = _parse_labels(file)
        img_path = os.path.join(test_dir, file)
        
        # Combine features exactly as done in training
        features = extract_short_features(img_path) + extract_short_features(getDAPI(img_path))
        
        # Predict and check against ground truth
        predicted_mag = clf.predict([features])[0]
        if true_mag not in error_dict.keys():
            error_dict[true_mag] = (0,0)
        
        curr_right, curr_total = error_dict[true_mag]
        if predicted_mag != true_mag:
            errors += 1
        else:
            curr_right += 1
        curr_total += 1
        error_dict[true_mag] = (curr_right, curr_total)
        total += 1
    
    print(error_dict)
    return errors / total if total > 0 else 0.0

def get_treatment_error_rate(test_dir="data/test"):
    # Load model once to save time
    model_path = os.path.join(MODELS_DIR, 'treatment_model.pkl')
    if not os.path.exists(model_path):
        train_treatment()
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
        
    errors, total = 0, 0
    error_dict = dict()
    
    for file in tqdm(os.listdir(test_dir)):
        # Skip blacklisted files and DAPI images (since we process them in pairs)
        if file in blacklist or "TRANS" not in file:
            continue
        if(np.random.rand() < cull_rate):
            continue
            
        _, _, true_treatment = _parse_labels(file)
        img_path = os.path.join(test_dir, file)
        
        # Combine features exactly as done in training
        features = extract_features(img_path) + extract_features(getDAPI(img_path))
        
        # Predict and check against ground truth
        predicted_treatment = clf.predict([features])[0]
        if true_treatment not in error_dict.keys():
            error_dict[true_treatment] = (0,0)
        
        curr_right, curr_total = error_dict[true_treatment]
        if predicted_treatment != true_treatment:
            errors += 1
        else:
            curr_right += 1
        curr_total += 1
        error_dict[true_treatment] = (curr_right, curr_total)
        total += 1
    
    print(error_dict)
    return errors / total if total > 0 else 0.0

print(get_treatment_error_rate())

# print(predict_group("data/test/G7_20x_D3_F1_T4_TRANS.jpg"))
