import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    ConfusionMatrixDisplay,
    classification_report, 
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score, 
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
    brier_score_loss,
    log_loss
)
from scipy import stats
import json
from collections import Counter, defaultdict
import plotly.graph_objects as go

class ViolenceDetectionEvaluator:
    """
    Evaluation metrics for violence detection results.
    """
    
    def __init__(self, results_df: pd.DataFrame, ground_truth_col: str = 'label'):
        """
        Initialize evaluator with results dataframe.
        
        Args:
            results_df: DataFrame with columns:
                - video_name
                - violence_probability (0-1)
                - confidence (0-1)
                - abstain (bool)
                - rationale
                - primary_modalities (e.g., "transcripts|audio|vision")
                - primary_models (e.g., "qwen|whisper|clap|beats|clip|xclip")
                - ground_truth (0/1)
        """
        self.df = results_df.copy()
        self.ground_truth_col = ground_truth_col
        
        # Parse modalities and models
        self._parse_primary_attributes()
        
    
    def _parse_primary_attributes(self):
        """Parse primary_modalities and primary_models columns."""
        # Split modalities
        self.df['modalities_list'] = self.df['primary_modalities'].apply(
            lambda x: x.split('|') if pd.notna(x) else []
        )
        self.df['top_modality'] = self.df['modalities_list'].apply(
            lambda x: x[0] if len(x) > 0 else None
        )
        
        if 'primary_models' in self.df.columns:
            # Split models
            self.df['models_list'] = self.df['primary_models'].apply(
                lambda x: x.split('|') if pd.notna(x) else []
            )
            self.df['top_model'] = self.df['models_list'].apply(
                lambda x: x[0] if len(x) > 0 else None
            )
    
    def evaluate_with_threshold(self, threshold: float = 0.5, confidence_filter: float = None) -> dict:
        """
        Evaluate binary classification metrics at a fixed threshold.
        
        Args:
            threshold: Probability threshold for classifying as violence (default: 0.5)
            confidence_filter: If set, only evaluate samples with confidence >= this value.
        
        Returns:
            dict: Evaluation metrics
        """
        # Create binary predictions
        self.df['predicted_binary'] = ((self.df['violence_probability'] >= threshold) & (self.df['confidence'] >= threshold)).astype(int)
        
        y_true = self.df[self.ground_truth_col]
        y_pred = self.df['predicted_binary']
        y_prob = self.df['violence_probability']
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        metrics = {
            'threshold': threshold,
            'total_samples': len(y_true),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'f1_score': f1_score(y_true, y_pred),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred),
            'balanced_accuracy': ((tp / (tp + fn)) + (tn / (tn + fp))) / 2 if (tp + fn) > 0 and (tn + fp) > 0 else 0,
        }
        
        # Add classification report
        report = classification_report(y_true, y_pred, output_dict=True, target_names=['Non-Violence', 'Violence'])
        metrics['classification_report'] = report
        
        return metrics
    
    def find_optimal_threshold(self, metric: str = 'f1'):
        """
        Find optimal threshold by maximizing a metric.
        
        Args:
            metric: 'f1', 'accuracy', 'balanced_accuracy', 'mcc'
        
        Returns:
            dict: {optimal_threshold, metric_value, all_results}
        """
        thresholds = np.arange(0.1, 1.01, 0.01)
        results = []
        
        for thresh in thresholds:
            metrics = self.evaluate_with_threshold(thresh)
            results.append({
                'threshold': thresh,
                'f1_score': metrics['f1_score'],
                'accuracy': metrics['accuracy'],
                'balanced_accuracy': metrics['balanced_accuracy'],
                'mcc': metrics['mcc'],
                'precision': metrics['precision'],
                'recall': metrics['recall']
            })
        
        results_df = pd.DataFrame(results)
        optimal_idx = results_df[metric].idxmax()
        optimal_thresh = results_df.loc[optimal_idx, 'threshold']
        
        return {
            'optimal_threshold': optimal_thresh,
            'metric_value': results_df.loc[optimal_idx, metric],
            'all_results': results_df
        }
    
    def analyze_modality_combinations(self, top_n: int = 3, threshold: float = 0.5):
        """
        Analyze performance of different modality combinations.
        
        Args:
            top_n: Number of top modalities to consider
            threshold: Probability threshold
        
        Returns:
            pd.DataFrame: Performance by modality combination
        """
        self.df['predicted_binary'] = (self.df['violence_probability'] >= threshold).astype(int)
        self.df['correct'] = self.df['predicted_binary'] == self.df[self.ground_truth_col]
        self.df['incorrect'] = self.df['predicted_binary'] != self.df[self.ground_truth_col]
        
        # Get top N modalities for each video
        self.df[f'top_{top_n}_modalities'] = self.df['modalities_list'].apply(
            lambda x: '|'.join(x[:top_n]) if len(x) > 0 else 'none'
        )
        
        # Group by modality combination
        combo_stats = self.df.groupby(f'top_{top_n}_modalities').agg({
            'correct': ['count', 'sum'],
            'incorrect': 'sum',
        }).round(3)
        
        combo_stats.columns = ['total_videos', 'correct_predictions', 'incorrect_predictions']
        combo_stats = combo_stats.sort_values('total_videos', ascending=False)
        
        return combo_stats
    
    def plot_classification_report(self, metrics, save_path: str = None):
        """Plot classification report """
        # Extract classification report
        report_data = {k: v for k, v in metrics['classification_report'].items() 
                    if k not in ['accuracy', 'macro avg', 'weighted avg']}

        # Convert to DataFrame
        df_report = pd.DataFrame(report_data).T
        df_report = df_report[['precision', 'recall', 'f1-score']]

        # Create plotly heatmap
        fig = go.Figure(data=go.Heatmap(
            z=df_report.values,
            x=df_report.columns,
            y=df_report.index,
            text=df_report.values.round(3),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorscale='Blues',
            colorbar=dict(title="Score")
        ))

        fig.update_layout(
            title='Classification Report Heatmap',
            xaxis_title='Metrics',
            yaxis_title='Class',
            width=800,
            height=400
        )

        fig.show()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    
    def plot_confusion_matrix(self, metrics, save_path: str = None):
        """Plot confusion matrix."""
        # Confusion matrix
        cm = np.array([
            [metrics['true_negatives'], metrics['false_positives']],
            [metrics['false_negatives'], metrics['true_positives']]
        ])

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()
    
    def plot_threshold_analysis(self, save_path: str = None):
        """Plot how metrics vary with threshold."""
        optimal = self.find_optimal_threshold('f1_score')
        results_df = optimal['all_results']
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        # F1, Precision, Recall
        ax = axes[0]
        ax.plot(results_df['threshold'], results_df['f1_score'], label='F1', linewidth=2)
        ax.plot(results_df['threshold'], results_df['precision'], label='Precision', linewidth=2)
        ax.plot(results_df['threshold'], results_df['recall'], label='Recall', linewidth=2)
        ax.axvline(optimal['optimal_threshold'], color='red', linestyle='--', label=f'Optimal={optimal["optimal_threshold"]:.2f}')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('F1, Precision, Recall vs Threshold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Accuracy & MCC
        ax = axes[1]
        ax.plot(results_df['threshold'], results_df['accuracy'], label='Accuracy', linewidth=2)
        ax.plot(results_df['threshold'], results_df['mcc'], label='MCC', linewidth=2)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Accuracy & MCC vs Threshold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()