import numpy as np
import cv2
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops

def extract_histogram(img, bins=(8, 8, 8)):
    # Extract RGB color histogram
    hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    
    # Normalize histogram to [0,1] range
    cv2.normalize(hist, hist)
    
    # Return flattened histogram
    return hist.flatten()

def extract_hog(img):
    # Convert to grayscale
    gray = rgb2gray(img)
    
    # Extract HOG features
    features, hog_image = hog(gray, 
                      orientations=9, 
                      pixels_per_cell=(16, 16),
                      cells_per_block=(2, 2), 
                      visualize=True, 
                      block_norm='L2-Hys')
    
    return features, hog_image

def extract_glcm(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate GLCM
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    # Extract GLCM properties
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    # Combine all GLCM features
    features = np.array([contrast, dissimilarity, homogeneity, asm, energy, correlation])
    
    return features

# def extract_color_stats(img):
#     # Convert to HSV for better color representation
#     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
#     # Extract value channel (brightness)
#     v_channel = img_hsv[:,:,2]
    
#     # Calculate statistics
#     mean = np.mean(v_channel)
#     variance = np.var(v_channel)
#     median = np.median(v_channel)
    
#     return np.array([mean, variance, median])

def extract_features(img):
    # Resize image to standard size
    img_resized = cv2.resize(img, (128, 128))
    
    # Extract features
    hist_features = extract_histogram(img_resized)
    hog_features, _ = extract_hog(img_resized)
    glcm_features = extract_glcm(img_resized)
    # color_stats = extract_color_stats(img_resized)
    
    # Combine features
    combined_features = np.hstack([hist_features, hog_features, glcm_features])
    
    return combined_features 