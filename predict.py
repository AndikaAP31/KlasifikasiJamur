import os
import argparse
import glob
from utils import load_components, predict_image, visualize_prediction

def main():
    """Main function to predict mushroom classes from images"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict mushroom classes from images')
    parser.add_argument('--image_path', type=str, required=True, 
                        help='Path to image file or directory containing images')
    parser.add_argument('--model_dir', type=str, default='.', 
                        help='Directory containing model files')
    parser.add_argument('--visualize', action='store_true', 
                        help='Visualize predictions')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 50)
    print("Mushroom Classification Prediction")
    print("=" * 50)
    print(f"Image path: {args.image_path}")
    print(f"Model directory: {args.model_dir}")
    print(f"Visualize: {args.visualize}")
    print("=" * 50)
    
    # Load model components
    print("\nLoading model components...")
    components = load_components(args.model_dir)
    model = components['model']
    scaler = components['scaler']
    pca = components['pca']
    label_encoder = components['label_encoder']
    
    try:
        prediction, confidence = predict_image(
            args.image_path, model, scaler, pca, label_encoder
        )
            
        print(f"\nPrediction: {prediction}")
        print(f"Confidence: {confidence*100:.2f}%")
            
        if args.visualize:
            visualize_prediction(args.image_path, prediction, confidence)
                
    except Exception as e:
        print(f"Error processing image: {str(e)}")
    
    print("\nPrediction completed successfully!")

if __name__ == "__main__":
    main() 