import os
import argparse
import time
import matplotlib.pyplot as plt
from data_loader import MushroomDataLoader
from model_training import MushroomClassifier
from utils import save_components

def main():
    """Main function to train the mushroom classification model"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train mushroom classification model')
    parser.add_argument('--dataset_path', type=str, required=True, 
                        help='Path to the dataset directory')
    parser.add_argument('--max_images', type=int, default=1000, 
                        help='Maximum number of images per class')
    parser.add_argument('--test_size', type=float, default=0.2, 
                        help='Proportion of data to use for testing')
    parser.add_argument('--grid_search', action='store_true', 
                        help='Perform grid search for hyperparameter tuning')
    parser.add_argument('--output_dir', type=str, default='.', 
                        help='Directory to save model files')
    parser.add_argument('--no_pca', action='store_true', 
                        help='Disable PCA dimensionality reduction')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 50)
    print("Mushroom Classification Model Training")
    print("=" * 50)
    print(f"Dataset path: {args.dataset_path}")
    print(f"Max images per class: {args.max_images}")
    print(f"Test size: {args.test_size}")
    print(f"Grid search: {args.grid_search}")
    print(f"PCA: {not args.no_pca}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 50)
    
    # Start timing
    start_time = time.time()
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    data_loader = MushroomDataLoader(args.dataset_path, args.max_images)
    data, labels, class_counts = data_loader.load_dataset()
    
    # Apply preprocessing (scaling and optionally PCA)
    processed_data = data_loader.preprocess_data(data, apply_pca=not args.no_pca)
    
    # Create classifier
    classifier = MushroomClassifier()
    
    # Split data
    print("\n2. Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = classifier.split_data(
        processed_data, labels, test_size=args.test_size
    )
    
    # Train model
    print("\n3. Training model...")
    if args.grid_search:
        # Train with grid search
        model, best_params = classifier.grid_search(X_train, y_train)
    else:
        # Train with default parameters
        model = classifier.train_svm(X_train, y_train)
    
    # Evaluate model and save visualizations
    print("\n4. Evaluating model...")
    accuracy, report, plot_paths = classifier.evaluate(
        X_test, y_test, 
        class_names=data_loader.label_encoder.classes_,
        plot_cm=False,  # Don't show plots during training
        save_plots=True,  # Save plots to disk
        output_dir=args.output_dir  # Save in the specified output directory
    )
    
    # Save model and components
    print("\n5. Saving model and components...")
    save_components(
        model=classifier.model,
        scaler=data_loader.scaler,
        pca=data_loader.pca,
        label_encoder=data_loader.label_encoder,
        base_path=args.output_dir
    )
    
    # Print summary
    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print("Training Summary")
    print("=" * 50)
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Final model accuracy: {accuracy*100:.2f}%")
    print(f"Class distribution: {class_counts}")
    
    # Print saved visualization paths
    if plot_paths:
        print("\nSaved visualizations:")
        for name, path in plot_paths.items():
            print(f"- {name}: {path}")
    
    # Print feature importance if using linear SVM
    if hasattr(classifier.model, 'kernel') and classifier.model.kernel == 'linear':
        print("\nTop feature importances:")
        importance = abs(classifier.model.coef_[0])
        top_indices = importance.argsort()[-10:][::-1]
        for i, idx in enumerate(top_indices):
            print(f"{i+1}. Feature {idx}: {importance[idx]:.4f}")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main() 