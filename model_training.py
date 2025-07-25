import numpy as np
import time
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class MushroomClassifier:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.training_time = None
        
    def split_data(self, X, y, test_size=0.2, verbose=True):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        if verbose:
            print(f"Training set: {len(X_train)} samples")
            print(f"Testing set: {len(X_test)} samples")
            
        return X_train, X_test, y_train, y_test
    
    def train_svm(self, X_train, y_train, kernel='rbf', C=100, gamma='scale', verbose=True):
        if verbose:
            print(f"\nTraining SVM model with {kernel} kernel...")
            start_time = time.time()
            
        # Create and train model
        self.model = svm.SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True,
            random_state=self.random_state
        )
        
        self.model.fit(X_train, y_train)
        
        if verbose:
            self.training_time = time.time() - start_time
            print(f"Model training completed in {self.training_time:.2f} seconds")
            
        return self.model
    
    def grid_search(self, X_train, y_train, param_grid=None, cv=5, verbose=True):
        if param_grid is None:
            param_grid = [
                {
                    'kernel': ['rbf'],
                    'C': [0.01, 0.1, 1, 10, 100, 1000],
                    'gamma': ['scale', 'auto', 0.01, 0.1]
                }                
            ]
            
        if verbose:
            print("\n🔍 Performing hyperparameter tuning with GridSearchCV...")
            start_time = time.time()
            
        # Create grid search
        grid_search = GridSearchCV(
            estimator=svm.SVC(probability=True, random_state=self.random_state),
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1 if verbose else 0
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model and parameters
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        if verbose:
            self.training_time = time.time() - start_time
            print(f"\n✅ Grid search completed in {self.training_time:.2f} seconds")
            print(f"🔧 Best parameters: {self.best_params}")
            print(f"🎯 Best cross-validation score: {grid_search.best_score_:.4f}")
            
        return self.model, self.best_params
    
    def evaluate(self, X_test, y_test, class_names=None, verbose=True, plot_cm=True, save_plots=False, output_dir='.'):
        if self.model is None:
            raise ValueError("Model not trained. Call train_svm or grid_search first.")
            
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        if verbose:
            print(f"\nTest Accuracy: {accuracy*100:.2f}%")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=class_names))
        
        plot_paths = {}
        
        # Plot confusion matrix
        if plot_cm or save_plots:
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Prediksi')
            plt.ylabel('Aktual')
            plt.title('Confusion Matrix')
            
            if save_plots:
                # Create directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Save confusion matrix
                cm_path = os.path.join(output_dir, 'confusion_matrix.png')
                plt.savefig(cm_path, bbox_inches='tight', dpi=300)
                plot_paths['confusion_matrix'] = cm_path
                print(f"Confusion matrix saved to {cm_path}")
            
            if plot_cm:
                plt.show()
            else:
                plt.close()
        
        # Create classification report visualization
        if save_plots:
            self._save_classification_report_plot(report, class_names, output_dir)
            report_path = os.path.join(output_dir, 'classification_report.png')
            plot_paths['classification_report'] = report_path
            
        return accuracy, report, plot_paths if save_plots else None
    
    def _save_classification_report_plot(self, report, class_names, output_dir):
        
        # Extract metrics for each class
        metrics = ['precision', 'recall', 'f1-score']
        class_data = {}
        
        for cls in class_names:
            class_data[cls] = [report[cls][metric] for metric in metrics]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        x = np.arange(len(metrics))
        width = 0.35
        
        # Plot bars for each class
        for i, (cls, values) in enumerate(class_data.items()):
            offset = width * (i - 0.5)
            bars = plt.bar(x + offset, values, width, label=cls)
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom')
        
        # Add labels and legend
        plt.ylabel('Skor')
        plt.title('Laporan Klasifikasi')
        plt.xticks(x, metrics)
        plt.ylim(0, 1.15)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        report_path = os.path.join(output_dir, 'classification_report.png')
        plt.savefig(report_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Classification report visualization saved to {report_path}")
        return report_path
    
    def save_model(self, model_path='best_mushroom_classifier.joblib'):
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
            
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path='best_mushroom_classifier.joblib'):
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        
        return self.model
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained. Call train_svm or grid_search first.")
            
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities 