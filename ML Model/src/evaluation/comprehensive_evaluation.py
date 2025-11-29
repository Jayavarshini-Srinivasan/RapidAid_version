"""
Comprehensive Model Evaluation Module
Implements evaluation for both regression and classification models:
- Regression: MSE, RMSE, MAE, R² score
- Classification: Precision, Recall, F1-score, Accuracy
- Confusion matrices with visualization
- K-fold cross-validation
- Model comparison visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ComprehensiveEvaluator:
    """Comprehensive model evaluation with metrics and visualizations"""
    
    def __init__(self, task_type='classification'):
        """
        Initialize the ComprehensiveEvaluator
        
        Args:
            task_type: 'classification' or 'regression'
        """
        self.task_type = task_type
        self.evaluation_results = {}
        
    # ==================== REGRESSION EVALUATION ====================
    
    def evaluate_regression(self, y_true, y_pred, model_name='Model'):
        """
        Evaluate regression model with comprehensive metrics
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            model_name: Name of the model for reporting
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100  # Mean Absolute Percentage Error
        median_ae = np.median(np.abs(y_true - y_pred))
        
        results = {
            'model_name': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'median_ae': median_ae
        }
        
        print("\n" + "="*60)
        print(f"REGRESSION EVALUATION: {model_name}")
        print("="*60)
        print(f"Mean Squared Error (MSE):     {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE):     {mae:.4f}")
        print(f"R² Score:                      {r2:.4f}")
        print(f"Mean Absolute % Error (MAPE):  {mape:.2f}%")
        print(f"Median Absolute Error:         {median_ae:.4f}")
        print("="*60)
        
        self.evaluation_results[model_name] = results
        return results
    
    # ==================== CLASSIFICATION EVALUATION ====================
    
    def evaluate_classification(self, y_true, y_pred, y_pred_proba=None, 
                               model_name='Model', class_names=None):
        """
        Evaluate classification model with comprehensive metrics
        
        Args:
            y_true: True target labels
            y_pred: Predicted target labels
            y_pred_proba: Predicted probabilities (optional)
            model_name: Name of the model for reporting
            class_names: List of class names for reporting
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate ROC-AUC if probabilities are available
        roc_auc = None
        if y_pred_proba is not None:
            try:
                # Check if multi-class
                if len(np.unique(y_true)) > 2 or y_pred_proba.shape[1] > 2:
                    roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                else:
                    # Binary case - use probability of positive class
                    if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] == 2:
                        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                    else:
                        roc_auc = roc_auc_score(y_true, y_pred_proba)
            except Exception as e:
                print(f"Could not calculate ROC-AUC: {e}")

        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm
        }
        
        print("\n" + "="*60)
        print(f"CLASSIFICATION EVALUATION: {model_name}")
        print("="*60)
        print(f"Accuracy:   {accuracy:.4f}")
        print(f"Precision:  {precision:.4f}")
        print(f"Recall:     {recall:.4f}")
        print(f"F1-Score:   {f1:.4f}")
        if roc_auc is not None:
            print(f"ROC-AUC:    {roc_auc:.4f}")
        print("\nPer-Class Metrics:")
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(precision_per_class))]
        for i, (p, r, f, name) in enumerate(zip(precision_per_class, recall_per_class, 
                                                f1_per_class, class_names)):
            print(f"  {name}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}")
        print("="*60)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
        
        self.evaluation_results[model_name] = results
        return results
    
    # ==================== CONFUSION MATRIX VISUALIZATION ====================
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name='Model', 
                            class_names=None, save_path=None):
        """
        Plot confusion matrix with visualization
        
        Args:
            y_true: True target labels
            y_pred: Predicted target labels
            model_name: Name of the model
            class_names: List of class names
            save_path: Path to save the figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved confusion matrix to {save_path}")
        
        plt.show()

    # ==================== ROC CURVE VISUALIZATION ====================

    def plot_roc_curve(self, y_true, y_pred_proba, model_name='Model', 
                      class_names=None, save_path=None):
        """
        Plot ROC curve (supports binary and multi-class)
        
        Args:
            y_true: True target labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            class_names: List of class names
            save_path: Path to save the figure
        """
        if y_pred_proba is None:
            print("No probabilities provided. Cannot plot ROC curve.")
            return

        n_classes = y_pred_proba.shape[1]
        
        # Binarize the output for multi-class
        if n_classes > 2:
            # Ensure y_true is numeric
            if y_true.dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_true = le.fit_transform(y_true)
                
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
        else:
            y_true_bin = y_true
            
        plt.figure(figsize=(10, 8))
        
        if n_classes > 2:
            # Multi-class ROC
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            plt.plot(fpr["micro"], tpr["micro"],
                     label='micro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["micro"]),
                     color='deeppink', linestyle=':', linewidth=4)
            
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
            
            if class_names is None:
                class_names = [f'Class {i}' for i in range(n_classes)]
                
            for i, color in zip(range(n_classes), colors):
                if i < len(class_names):
                    label = f'ROC curve of {class_names[i]} (area = {roc_auc[i]:0.2f})'
                else:
                    label = f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})'
                    
                plt.plot(fpr[i], tpr[i], color=color, lw=2, label=label)
                
        else:
            # Binary ROC
            # Assuming y_pred_proba has 2 columns, use the second one (positive class)
            if y_pred_proba.shape[1] == 2:
                probs = y_pred_proba[:, 1]
            else:
                probs = y_pred_proba
                
            fpr, tpr, _ = roc_curve(y_true, probs)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC curve (area = {roc_auc:.2f})')
            
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'Receiver Operating Characteristic (ROC) - {model_name}', 
                 fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved ROC curve to {save_path}")
        
        plt.show()
    
    # ==================== CROSS-VALIDATION ====================
    
    def cross_validate(self, model, X, y, cv=5, scoring=None, task_type=None):
        """
        Perform k-fold cross-validation
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target vector
            cv: Number of folds
            scoring: Scoring metric (if None, uses default for task type)
            task_type: 'classification' or 'regression' (if None, uses self.task_type)
        
        Returns:
            Cross-validation scores
        """
        if task_type is None:
            task_type = self.task_type
        
        # Select appropriate CV strategy and scoring
        if task_type == 'classification':
            if scoring is None:
                scoring = 'f1_macro'
            cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            if scoring is None:
                scoring = 'neg_mean_squared_error'
            cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        print(f"\nPerforming {cv}-fold cross-validation...")
        cv_scores = cross_val_score(model, X, y, cv=cv_strategy, 
                                   scoring=scoring, n_jobs=-1)
        
        print(f"\n{cv}-Fold Cross-Validation Results:")
        print(f"  Mean Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"  Min Score:  {cv_scores.min():.4f}")
        print(f"  Max Score:  {cv_scores.max():.4f}")
        print(f"  Individual fold scores: {cv_scores}")
        
        return cv_scores
    
    # ==================== MODEL COMPARISON VISUALIZATIONS ====================
    
    def compare_models(self, save_path=None):
        """
        Create comparison visualizations for multiple models
        
        Args:
            save_path: Path to save the figure
        """
        if len(self.evaluation_results) == 0:
            print("No evaluation results available. Please evaluate models first.")
            return
        
        if self.task_type == 'classification':
            self._compare_classification_models(save_path)
        else:
            self._compare_regression_models(save_path)
    
    def _compare_classification_models(self, save_path=None):
        """Compare classification models"""
        models = list(self.evaluation_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            values = [self.evaluation_results[model][metric] for model in models]
            
            axes[idx].bar(models, values, color=sns.color_palette("husl", len(models)))
            axes[idx].set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            axes[idx].set_ylabel('Score', fontsize=12)
            axes[idx].set_ylim([0, 1])
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('Model Comparison - Classification Metrics', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved comparison plot to {save_path}")
        
        plt.show()
    
    def _compare_regression_models(self, save_path=None):
        """Compare regression models"""
        models = list(self.evaluation_results.keys())
        metrics = ['mse', 'rmse', 'mae', 'r2_score']
        metric_labels = ['MSE', 'RMSE', 'MAE', 'R² Score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [self.evaluation_results[model][metric] for model in models]
            
            # For R², higher is better; for others, lower is better
            color = 'green' if metric == 'r2_score' else 'red'
            axes[idx].bar(models, values, color=color, alpha=0.7)
            axes[idx].set_title(f'{label}', fontsize=14, fontweight='bold')
            axes[idx].set_ylabel('Score', fontsize=12)
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[idx].text(i, v + max(values) * 0.02, f'{v:.3f}', 
                             ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('Model Comparison - Regression Metrics', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved comparison plot to {save_path}")
        
        plt.show()
    
    # ==================== PREDICTION VS ACTUAL PLOTS ====================
    
    def plot_predictions_vs_actual(self, y_true, y_pred, model_name='Model', 
                                  save_path=None):
        """
        Plot predictions vs actual values (for regression)
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            model_name: Name of the model
            save_path: Path to save the figure
        """
        if self.task_type != 'regression':
            print("This plot is only for regression tasks.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot: predictions vs actual
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                    'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Values', fontsize=12)
        axes[0].set_ylabel('Predicted Values', fontsize=12)
        axes[0].set_title(f'Predictions vs Actual - {model_name}', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Residual plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Values', fontsize=12)
        axes[1].set_ylabel('Residuals', fontsize=12)
        axes[1].set_title(f'Residual Plot - {model_name}', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved prediction plot to {save_path}")
        
        plt.show()
    
    # ==================== FEATURE IMPORTANCE VISUALIZATION ====================
    
    def plot_feature_importance(self, model, feature_names, top_n=20, 
                               model_name='Model', save_path=None):
        """
        Plot feature importance
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            top_n: Number of top features to display
            model_name: Name of the model
            save_path: Path to save the figure
        """
        if not hasattr(model, 'feature_importances_'):
            print("Model does not have feature_importances_ attribute.")
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices], color='steelblue')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances - {model_name}', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved feature importance plot to {save_path}")
        
        plt.show()
    
    # ==================== SUMMARY REPORT ====================
    
    def generate_summary_report(self, save_path=None):
        """
        Generate a summary report of all evaluations
        
        Args:
            save_path: Path to save the report (CSV or JSON)
        """
        if len(self.evaluation_results) == 0:
            print("No evaluation results available.")
            return
        
        # Convert to DataFrame
        if self.task_type == 'classification':
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        else:
            metrics = ['mse', 'rmse', 'mae', 'r2_score']
        
        report_data = []
        for model_name, results in self.evaluation_results.items():
            row = {'model': model_name}
            for metric in metrics:
                row[metric] = results.get(metric, np.nan)
            report_data.append(row)
        
        df_report = pd.DataFrame(report_data)
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY REPORT")
        print("="*60)
        print(df_report.to_string(index=False))
        print("="*60)
        
        if save_path:
            if save_path.endswith('.csv'):
                df_report.to_csv(save_path, index=False)
            elif save_path.endswith('.json'):
                df_report.to_json(save_path, orient='records', indent=2)
            print(f"✓ Saved summary report to {save_path}")
        
        return df_report


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("Comprehensive Model Evaluation Module")
    print("This module provides evaluation metrics and visualizations for ML models.")
    print("\nUsage:")
    print("  from comprehensive_evaluation import ComprehensiveEvaluator")
    print("  evaluator = ComprehensiveEvaluator(task_type='classification')")
    print("  results = evaluator.evaluate_classification(y_true, y_pred)")

