import os
import joblib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from feature_extraction import extract_features

def save_components(model, scaler, pca, label_encoder, base_path='.'):
    """
    Save all model components to disk
    
    Args:
        model: Trained model
        scaler: Fitted StandardScaler
        pca: Fitted PCA transformer
        label_encoder: Fitted LabelEncoder
        base_path: Directory to save the components
        
    Returns:
        paths: Dictionary of saved file paths
    """
    # Create paths
    model_path = os.path.join(base_path, 'best_mushroom_classifier.joblib')
    scaler_path = os.path.join(base_path, 'scaler.joblib')
    pca_path = os.path.join(base_path, 'pca.joblib')
    le_path = os.path.join(base_path, 'label_encoder.joblib')
    
    # Save components
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(pca, pca_path)
    joblib.dump(label_encoder, le_path)
    
    print("All components saved successfully!")
    print(f"Model: {model_path}")
    print(f"Scaler: {scaler_path}")
    print(f"PCA: {pca_path}")
    print(f"Label Encoder: {le_path}")
    
    return {
        'model': model_path,
        'scaler': scaler_path,
        'pca': pca_path,
        'label_encoder': le_path
    }

def load_components(base_path='.'):
    """
    Load all model components from disk
    
    Args:
        base_path: Directory where components are saved
        
    Returns:
        components: Dictionary of loaded components
    """
    # Create paths
    model_path = os.path.join(base_path, 'best_mushroom_classifier.joblib')
    scaler_path = os.path.join(base_path, 'scaler.joblib')
    pca_path = os.path.join(base_path, 'pca.joblib')
    le_path = os.path.join(base_path, 'label_encoder.joblib')
    
    # Check if files exist
    files = [model_path, scaler_path, pca_path, le_path]
    missing = [f for f in files if not os.path.exists(f)]
    
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")
    
    # Load components
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    label_encoder = joblib.load(le_path)
    
    print("All components loaded successfully!")
    
    return {
        'model': model,
        'scaler': scaler,
        'pca': pca,
        'label_encoder': label_encoder
    }

def predict_image(image_path, model, scaler, pca, label_encoder):
    """
    Predict the class of a mushroom image
    
    Args:
        image_path: Path to the image file
        model: Trained classifier model
        scaler: Fitted StandardScaler
        pca: Fitted PCA transformer
        label_encoder: Fitted LabelEncoder
        
    Returns:
        prediction: Class label
        confidence: Prediction confidence (probability)
    """
    # Load image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Extract features
    features = extract_features(img)
    
    # Reshape for single sample
    features = features.reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Apply PCA
    features_pca = pca.transform(features_scaled)
    
    # Predict
    pred = model.predict(features_pca)
    pred_label = label_encoder.inverse_transform(pred)
    
    # Get probability
    prob = model.predict_proba(features_pca)
    confidence = np.max(prob)
    
    return pred_label[0], confidence

def visualize_prediction(image_path, prediction, confidence):
 
    # Load and convert image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure
    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    
    # Add title with prediction and confidence
    title = f"Prediction: {prediction} (Confidence: {confidence*100:.2f}%)"
    plt.title(title)
    
    # Remove axes
    plt.axis('off')
    
    # Show plot
    plt.show()

def count_dataset_samples(base_dir):

    classes = ['Beracun', 'Tidak Beracun']
    counts = {}
    
    for label in classes:
        folder_path = os.path.join(base_dir, label)
        
        if not os.path.exists(folder_path):
            counts[label] = 0
            continue
            
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        counts[label] = len(image_files)
    
    return counts 