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
    
    # Check if input is a directory or a single file
    if os.path.isdir(args.image_path):
        # Process all images in directory
        image_files = []
        for ext in ['jpg', 'jpeg', 'png']:
            image_files.extend(glob.glob(os.path.join(args.image_path, f'*.{ext}')))
            image_files.extend(glob.glob(os.path.join(args.image_path, f'*.{ext.upper()}')))
        
        print(f"\nFound {len(image_files)} images in directory")
        
        # Process each image
        results = []
        for img_path in image_files:
            try:
                prediction, confidence = predict_image(
                    img_path, model, scaler, pca, label_encoder
                )
                
                results.append({
                    'image': os.path.basename(img_path),
                    'prediction': prediction,
                    'confidence': confidence
                })
                
                print(f"Image: {os.path.basename(img_path)}, "
                      f"Prediction: {prediction}, "
                      f"Confidence: {confidence*100:.2f}%")
                
                if args.visualize:
                    visualize_prediction(img_path, prediction, confidence)
            
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
        
        # Print summary
        print("\n" + "=" * 50)
        print("Prediction Summary")
        print("=" * 50)
        
        # Count predictions by class
        class_counts = {}
        for result in results:
            pred = result['prediction']
            if pred not in class_counts:
                class_counts[pred] = 0
            class_counts[pred] += 1
        
        for cls, count in class_counts.items():
            percentage = (count / len(results)) * 100
            print(f"{cls}: {count} images ({percentage:.1f}%)")
        
    else:
        # Process single image
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