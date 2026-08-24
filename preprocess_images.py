import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from utils.general import get_rally_dirs, HEIGHT, WIDTH, IMG_FORMAT

# ==============================================================================
# Configuration
# ==============================================================================
SOURCE_DIR = 'data'
TARGET_DIR = f'data_{HEIGHT}x{WIDTH}'
NUM_WORKERS = min(os.cpu_count(), 12)  # Adjust based on your CPU
# ==============================================================================

def process_single_rally(rally_rel_path):
    """Worker function to process one rally"""
    src_rally_path = os.path.join(SOURCE_DIR, rally_rel_path)
    tgt_rally_path = os.path.join(TARGET_DIR, rally_rel_path)
    
    if not os.path.exists(tgt_rally_path):
        os.makedirs(tgt_rally_path, exist_ok=True)
    
    # 1. Process frames
    frame_files = [f for f in os.listdir(src_rally_path) if f.endswith(IMG_FORMAT)]
    frames_for_median = []
    
    for f_name in frame_files:
        src_f = os.path.join(src_rally_path, f_name)
        tgt_f = os.path.join(tgt_rally_path, f_name)
        
        # Skip if already exists
        if os.path.exists(tgt_f):
            # If we need median, we might still need to read it, but let's assume we can skip
            # unless median.npz is missing
            if os.path.exists(os.path.join(tgt_rally_path, 'median.npz')):
                continue
        
        img = cv2.imread(src_f)
        if img is None: continue
        
        # Resize
        resized_img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        cv2.imwrite(tgt_f, resized_img)
        
        # Collect for median (convert BGR to RGB to match original logic)
        frames_for_median.append(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))

    # 2. Process Rally Median
    tgt_median_path = os.path.join(tgt_rally_path, 'median.npz')
    if not os.path.exists(tgt_median_path) and frames_for_median:
        median = np.median(np.array(frames_for_median), axis=0)
        np.savez(tgt_median_path, median=median)
        
    return len(frame_files)

def process_match_median(match_info):
    """Worker function for match-level medians"""
    src_match_path, tgt_match_path = match_info
    
    src_m_median = os.path.join(src_match_path, 'median.npz')
    tgt_m_median = os.path.join(tgt_match_path, 'median.npz')
    
    if os.path.exists(src_m_median) and not os.path.exists(tgt_m_median):
        if not os.path.exists(tgt_match_path):
            os.makedirs(tgt_match_path, exist_ok=True)
        median_img = np.load(src_m_median)['median']
        resized_median = cv2.resize(median_img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        np.savez(tgt_m_median, median=resized_median)

def preprocess():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    all_rallies = []
    match_medians_to_process = []

    for split in ['train', 'val', 'test']:
        split_src = os.path.join(SOURCE_DIR, split)
        if not os.path.exists(split_src):
            continue
            
        rally_dirs = get_rally_dirs(SOURCE_DIR, split)
        all_rallies.extend(rally_dirs)
        
        # Prepare match medians
        match_dirs = os.listdir(split_src)
        for m_dir in match_dirs:
            src_path = os.path.join(split_src, m_dir)
            tgt_path = os.path.join(TARGET_DIR, split, m_dir)
            if os.path.isdir(src_path):
                match_medians_to_process.append((src_path, tgt_path))

    print(f'Starting multi-process preprocessing with {NUM_WORKERS} workers...')
    print(f'Total rallies to process: {len(all_rallies)}')

    # 2. Process all rallies in parallel
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(process_single_rally, all_rallies), total=len(all_rallies), desc="Resizing Rallies"))

    # 3. Copy CSV files and resize match medians
    print("Copying CSV files and processing match medians...")
    for split in ['train', 'val', 'test']:
        split_src = os.path.join(SOURCE_DIR, split)
        if not os.path.exists(split_src): continue
        
        match_dirs = os.listdir(split_src)
        for m_dir in match_dirs:
            src_match_path = os.path.join(split_src, m_dir)
            tgt_match_path = os.path.join(TARGET_DIR, split, m_dir)
            if not os.path.isdir(src_match_path): continue
            
            # Copy CSV directories
            for csv_dir in ['csv', 'corrected_csv']:
                src_csv = os.path.join(src_match_path, csv_dir)
                tgt_csv = os.path.join(tgt_match_path, csv_dir)
                if os.path.exists(src_csv) and not os.path.exists(tgt_csv):
                    shutil.copytree(src_csv, tgt_csv)

    # 4. Process match medians in parallel
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(process_match_median, match_medians_to_process), total=len(match_medians_to_process), desc="Resizing Match Medians"))

    # 5. Generate correct img_config files for the target directory
    print("Generating correct img_config files based on source dimensions...")
    for split in ['train', 'val', 'test']:
        split_src = os.path.join(SOURCE_DIR, split)
        if not os.path.exists(split_src): continue
        
        rally_dirs = get_rally_dirs(SOURCE_DIR, split)
        img_scaler = []
        img_shape = []
        
        for rally_rel in tqdm(rally_dirs, desc=f"Config for {split}"):
            src_frame0 = os.path.join(SOURCE_DIR, rally_rel, f'0.{IMG_FORMAT}')
            if os.path.exists(src_frame0):
                # Get ORIGINAL dimensions
                img = cv2.imread(src_frame0)
                h, w = img.shape[:2]
                w_scaler, h_scaler = w / WIDTH, h / HEIGHT
                img_scaler.append((w_scaler, h_scaler))
                img_shape.append((w, h))
            else:
                img_scaler.append((1.0, 1.0))
                img_shape.append((WIDTH, HEIGHT))
        
        config_file = os.path.join(TARGET_DIR, f'img_config_{HEIGHT}x{WIDTH}_{split}.npz')
        np.savez(config_file, img_scaler=img_scaler, img_shape=img_shape)
        print(f"Saved {config_file}")

    print('\nPreprocessing complete!')

if __name__ == '__main__':
    preprocess()
