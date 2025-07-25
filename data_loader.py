import os
import numpy as np
import cv2
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from feature_extraction import extract_features

class MushroomDataLoader:
   
    def __init__(self, base_dir, max_images_per_class=1000):
        self.base_dir = base_dir
        self.max_images_per_class = max_images_per_class
        self.classes = ['Beracun', 'Tidak Beracun']
        self.scaler = StandardScaler()
        self.pca = None
        self.label_encoder = LabelEncoder()
        
    def load_dataset(self, verbose=True):
        
        if verbose:
            print("Loading and extracting features from images...")
            start_time = time.time()
        
        combined_features = []
        labels = []
        class_counts = {}
        
        for label in self.classes:
            folder_path = os.path.join(self.base_dir, label)
            
            # Check if folder exists
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Class folder not found: {folder_path}")
            
            # Get image files
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            original_count = len(image_files)
            class_counts[label] = original_count
            
            if verbose:
                print(f"Found {original_count} images in class '{label}'")
            
            # Limit the number of images per class if needed
            if len(image_files) > self.max_images_per_class:
                import random
                random.seed(42)  # For reproducibility
                image_files = random.sample(image_files, self.max_images_per_class)
                
                if verbose:
                    print(f"Limiting {label} class to {self.max_images_per_class} images")
            
            if verbose:
                print(f"Processing {label} images...")
            
            # Process each image
            for i, img_name in enumerate(image_files):
                if verbose and i % 100 == 0:
                    print(f"  Progress: {i}/{len(image_files)} images processed")
                    
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                # Extract features
                features = extract_features(img)
                
                combined_features.append(features)
                labels.append(label)
            
            if verbose:
                print(f"Completed processing {len(image_files)} images for class '{label}'")
        
        # Convert to numpy arrays
        data = np.array(combined_features)
        labels = np.array(labels)
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        if verbose:
            print(f"\nDataset Information:")
            print(f"Total dataset size: {len(data)} samples")
            print(f"Dataset sizes:")
            
            for class_name, count in class_counts.items():
                used_count = min(count, self.max_images_per_class) if count > self.max_images_per_class else count
                final_count = np.sum(labels == class_name)
                print(f"  Class '{class_name}': {count} original images → {used_count} used → {final_count} final samples")
            
            print(f"Classes: {list(self.label_encoder.classes_)}")
            print(f"Feature vector length: {data.shape[1]}")
            print(f"Data loading and feature extraction completed in {time.time() - start_time:.2f} seconds")
        
        return data, labels_encoded, class_counts
    
    def preprocess_data(self, data, apply_pca=True, pca_components=0.95, verbose=True):
       
        # Scale features
        if verbose:
            print("\nNormalizing features...")
            
        data_scaled = self.scaler.fit_transform(data)
        
        if verbose:
            print("Feature normalization completed.")
        
        # Apply PCA if requested
        if apply_pca:
            if verbose:
                print("\nApplying PCA...")
                
            self.pca = PCA(n_components=pca_components)
            data_pca = self.pca.fit_transform(data_scaled)
            
            if verbose:
                print(f"Shape before PCA: {data_scaled.shape}")
                print(f"Shape after PCA: {data_pca.shape}")
                print(f"Explained variance ratio: {sum(self.pca.explained_variance_ratio_):.4f}")
            
            return data_pca
        else:
            return data_scaled
    
    def transform_new_data(self, features):
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call preprocess_data first.")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Apply PCA if available
        if self.pca is not None:
            features_pca = self.pca.transform(features_scaled)
            return features_pca
        else:
            return features_scaled 