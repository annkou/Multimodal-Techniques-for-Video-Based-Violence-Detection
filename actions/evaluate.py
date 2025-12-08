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
    log_loss,
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
        
    # Normalize modalities - group transcript/transcripts
    def _normalize_modalities(self, modalities_str):
        if pd.isna(modalities_str):
            return []
        modalities = modalities_str.split('|') if isinstance(modalities_str, str) else modalities_str
        # Normalize transcript/transcripts to 'transcripts'
        normalized = []
        for mod in modalities:
            mod = mod.strip().lower()
            if mod in ['transcript', 'transcripts']:
                normalized.append('transcripts')
            else:
                normalized.append(mod)
        return normalized

    def _parse_primary_attributes(self):
        """Parse primary_modalities and primary_models columns."""

        # Split modalities
        self.df['modalities_list'] = self.df['primary_modalities'].apply(self._normalize_modalities)
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
        display(combo_stats)
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
    
    def plot_roc_curve_analysis(self, save_path: str = None):
        """
        Plot ROC curve and calculate AUC-ROC score.
        AUC-ROC (Area Under Curve):
        - Measures the model's ability to distinguish between classes
        - Range: 0.5 (random) to 1.0 (perfect)
        - Interpretation:
          * 0.90-1.00: Excellent
          * 0.80-0.90: Good
          * 0.70-0.80: Fair
          * 0.60-0.70: Poor
          * 0.50-0.60: Fail (no better than random)
        
        Returns:
            dict: ROC analysis results
        """
        y_true = self.df[self.ground_truth_col]
        y_prob = self.df['violence_probability']
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_tpr = tpr[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        
        # Print results
        print(f"\n{'='*70}")
        print(f"ROC CURVE ANALYSIS")
        print(f"{'='*70}")
        print(f"\nAUC-ROC Score: {auc_score:.4f}")
        
        if auc_score >= 0.90:
            interpretation = "Excellent - Model has outstanding discrimination ability"
        elif auc_score >= 0.80:
            interpretation = "Good - Model performs well"
        elif auc_score >= 0.70:
            interpretation = "Fair - Model has acceptable performance"
        elif auc_score >= 0.60:
            interpretation = "Poor - Model needs improvement"
        else:
            interpretation = "Fail - Model performs no better than random guessing"
        
        print(f"   Interpretation: {interpretation}")
        print(f"\nOptimal Operating Point:")
        print(f"   Threshold: {optimal_threshold:.3f}")
        print(f"   True Positive Rate (Recall): {optimal_tpr:.3f}")
        print(f"   False Positive Rate: {optimal_fpr:.3f}")
        print(f"   Specificity: {1-optimal_fpr:.3f}")
        
        # Plot ROC curve
        fig, ax = plt.subplots(figsize=(7, 5))
        
        ax.plot(fpr, tpr, linewidth=1.5, label=f'ROC Curve (AUC = {auc_score:.3f})', color='#2E86AB')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.0, label='Random Classifier (AUC = 0.50)', alpha=0.6)
        
        # Mark optimal point
        ax.scatter(optimal_fpr, optimal_tpr, s=200, c='red', marker='*', 
                  label=f'Optimal Point (threshold={optimal_threshold:.2f})', zorder=5)
        
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10, fontweight='bold')
        ax.set_ylabel('True Positive Rate (Recall/Sensitivity)', fontsize=10, fontweight='bold')
        ax.set_title('ROC Curve - Violence Detection Performance', fontsize=10, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        
        # Add shaded area under curve
        ax.fill_between(fpr, tpr, alpha=0.2, color='#2E86AB')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'auc_roc': auc_score,
            'optimal_threshold': optimal_threshold,
            'optimal_tpr': optimal_tpr,
            'optimal_fpr': optimal_fpr,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
    
    def calculate_probability_metrics(self):
        """
        Calculate Log Loss and Brier Score to evaluate probability quality.
        
        BRIER SCORE:
        - Measures the mean squared difference between predicted probabilities and actual outcomes
        - Formula: mean((predicted_prob - actual)²)
        - Range: 0 (perfect) to 1 (worst)
        - Lower is better
        - Interpretation:
          * < 0.10: Excellent calibration
          * 0.10-0.20: Good calibration
          * 0.20-0.25: Fair calibration
          * > 0.25: Poor calibration
        - Example: Predict 0.7 for violence, actual=violence → (0.7-1)² = 0.09
        
        LOG LOSS (Cross-Entropy):
        - Penalizes confident wrong predictions more heavily than Brier Score
        - Formula: -mean(actual×log(predicted) + (1-actual)×log(1-predicted))
        - Range: 0 (perfect) to ∞
        - Lower is better
        - Interpretation:
          * < 0.30: Excellent
          * 0.30-0.50: Good
          * 0.50-0.70: Fair
          * > 0.70: Poor
        - Example: Predict 0.9 for violence, actual=non-violence → -log(0.1) = 2.30 (very bad!)
        
        KEY DIFFERENCES:
        1. Sensitivity to Confidence:
           - Log Loss heavily penalizes overconfident wrong predictions
           - Brier Score treats all errors more uniformly (squared vs logarithmic)
        
        2. Scale:
           - Brier Score: 0-1 scale, easier to interpret
           - Log Loss: unbounded, can be very large for confident mistakes
        
        Returns:
            dict: metrics and analysis
        """
        y_true = self.df[self.ground_truth_col]
        y_prob = self.df['violence_probability']
        
        # Calculate metrics
        brier = brier_score_loss(y_true, y_prob)
        logloss = log_loss(y_true, y_prob)
        
        # Analyze prediction distribution
        violence_cases = y_prob[y_true == 1]
        non_violence_cases = y_prob[y_true == 0]
        
        print(f"\n{'='*70}")
        print(f"PROBABILITY CALIBRATION METRICS")
        print(f"{'='*70}")
        
        print(f"\nBRIER SCORE: {brier:.4f}")
        if brier < 0.10:
            brier_interp = "Excellent - Predictions are well-calibrated"
        elif brier < 0.20:
            brier_interp = "Good - Predictions are reasonably calibrated"
        elif brier < 0.25:
            brier_interp = "Fair - Some calibration issues present"
        else:
            brier_interp = "Poor - Significant calibration problems"
        print(f"   Interpretation: {brier_interp}")
        
        print(f"\nLOG LOSS: {logloss:.4f}")
        if logloss < 0.30:
            log_interp = "Excellent - Very few overconfident mistakes"
        elif logloss < 0.50:
            log_interp = "Good - Acceptable confidence levels"
        elif logloss < 0.70:
            log_interp = "Fair - Some overconfident errors"
        else:
            log_interp = "Poor - Many overconfident wrong predictions"
        print(f"   Interpretation: {log_interp}")
        
        print(f"\nCOMPARISON:")
        print(f"   Brier Score treats all errors uniformly (squared difference)")
        print(f"   Log Loss heavily penalizes overconfident wrong predictions")
        
        # if logloss / brier > 3:
        #     print(f"\nWARNING: Log Loss is {logloss/brier:.1f}x larger than Brier Score")
        #     print(f"   This indicates your model is making overconfident mistakes!")
        
        print(f"\nPREDICTION DISTRIBUTION:")
        print(f"   Violence Cases (ground truth = 1):")
        print(f"      Mean probability: {violence_cases.mean():.3f}")
        print(f"      Median probability: {violence_cases.median():.3f}")
        print(f"      Std deviation: {violence_cases.std():.3f}")
        
        print(f"\n   Non-Violence Cases (ground truth = 0):")
        print(f"      Mean probability: {non_violence_cases.mean():.3f}")
        print(f"      Median probability: {non_violence_cases.median():.3f}")
        print(f"      Std deviation: {non_violence_cases.std():.3f}")
        
        # Plot probability distributions
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        # alpha=0.6  # 60% opacity, 40% transparency
        ax = axes[0]
        ax.hist(violence_cases, bins=20, alpha=0.6, label='Violence (ground truth)', 
                color='red', edgecolor='black')
        ax.hist(non_violence_cases, bins=20, alpha=0.6, label='Non-Violence (ground truth)', 
                color='blue', edgecolor='black')
        ax.axvline(0.5, color='green', linestyle='--', linewidth=2, label='Threshold=0.5')
        ax.set_xlabel('Predicted Violence Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Probability Distribution by True Class', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        # Box plot
        ax = axes[1]
        bp = ax.boxplot([non_violence_cases, violence_cases], 
                        labels=['Non-Violence\n(ground truth)', 'Violence\n(ground truth)'],
                        patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor('blue')
        bp['boxes'][1].set_facecolor('red')
        for box in bp['boxes']:
            box.set_alpha(0.6)
        ax.axhline(0.5, color='green', linestyle='--', linewidth=2, label='Threshold=0.5')
        ax.set_ylabel('Predicted Violence Probability', fontsize=12, fontweight='bold')
        ax.set_title('Probability Range by True Class', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'brier_score': brier,
            'log_loss': logloss,
            'violence_mean_prob': violence_cases.mean(),
            'non_violence_mean_prob': non_violence_cases.mean(),
        }
    
    def analyze_modality_performance_by_outcome(self, threshold: float = 0.5, top_n: int = 5):
        """
        Analyze which modalities are most common in different prediction outcomes.
        
        This helps identify:
        - Which modalities lead to correct violence detection (TP)
        - Which modalities cause false alarms (FP)
        - Which modalities miss violence (FN)
        - Which modalities correctly identify non-violence (TN)
        
        Args:
            threshold: Classification threshold
            top_n: Number of top modalities to display per outcome
        
        Returns:
            dict: Modality analysis for each outcome type
        """
        # Create predictions
        self.df['predicted_binary'] = (self.df['violence_probability'] >= threshold).astype(int)
        
        # Categorize outcomes
        self.df['outcome'] = 'TN'  # True Negative (default)
        self.df.loc[(self.df[self.ground_truth_col] == 1) & 
                   (self.df['predicted_binary'] == 1), 'outcome'] = 'TP'  # True Positive
        self.df.loc[(self.df[self.ground_truth_col] == 0) & 
                   (self.df['predicted_binary'] == 1), 'outcome'] = 'FP'  # False Positive
        self.df.loc[(self.df[self.ground_truth_col] == 1) & 
                   (self.df['predicted_binary'] == 0), 'outcome'] = 'FN'  # False Negative
        
        results = {}
        
        # Analyze each outcome type
        for outcome_type in ['TP', 'FP', 'FN', 'TN']:
            outcome_df = self.df[self.df['outcome'] == outcome_type]
            
            if len(outcome_df) == 0:
                continue
            
            # Count modalities
            modality_counter = Counter()
            for modalities in outcome_df['modalities_list']:
                modality_counter.update(modalities)
            
            # Get top modalities
            top_modalities = modality_counter.most_common(top_n)
            
            # Calculate statistics
            avg_prob = outcome_df['violence_probability'].mean()
            avg_conf = outcome_df['confidence'].mean()
            
            results[outcome_type] = {
                'count': len(outcome_df),
                'top_modalities': top_modalities,
                'avg_probability': avg_prob,
                'avg_confidence': avg_conf
            }
            
            # Print analysis
            print(f"\n{'─'*70}")
            if outcome_type == 'TP':
                print(f"TRUE POSITIVES ({len(outcome_df)} cases)")
                print(f"   Description: Correctly identified violence")
            elif outcome_type == 'FP':
                print(f"FALSE POSITIVES ({len(outcome_df)} cases)")
                print(f"   Description: Incorrectly flagged non-violence as violence")
            elif outcome_type == 'FN':
                print(f"FALSE NEGATIVES ({len(outcome_df)} cases)")
                print(f"   Description: Missed actual violence")
            else:  # TN
                print(f"TRUE NEGATIVES ({len(outcome_df)} cases)")
                print(f"   Description: Correctly identified non-violence")
            
            print(f"\n   Average Violence Probability: {avg_prob:.3f}")
            print(f"   Average Confidence: {avg_conf:.3f}")
        
        # Visual comparison
        self._plot_modality_comparison(results, threshold)
        
        return results
    
    def _plot_modality_comparison(self, results: dict, threshold: float):
        """Helper function to visualize modality performance across outcomes based on their ranking/order."""
        
        # Analyze modality positions (1st, 2nd, 3rd, etc.)
        position_analysis = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'positions': []}))
        
        for outcome in ['TP', 'TN', 'FP', 'FN']:
            if outcome not in results:
                continue
                
            outcome_df = self.df[self.df['outcome'] == outcome]
            
            for _, row in outcome_df.iterrows():
                modalities = row['modalities_list']
                for position, modality in enumerate(modalities, start=1):
                    position_analysis[outcome][modality]['count'] += 1
                    position_analysis[outcome][modality]['positions'].append(position)
        
        # Calculate statistics for each modality in each outcome
        stats_data = []
        for outcome in ['TP', 'TN', 'FP', 'FN']:
            if outcome not in results:
                continue
                
            for modality, data in position_analysis[outcome].items():
                avg_position = np.mean(data['positions'])
                median_position = np.median(data['positions'])
                first_position_count = sum(1 for p in data['positions'] if p == 1)
                first_position_pct = (first_position_count / data['count'] * 100) if data['count'] > 0 else 0
                
                stats_data.append({
                    'outcome': outcome,
                    'modality': modality,
                    'total_uses': data['count'],
                    'avg_position': avg_position,
                    'median_position': median_position,
                    'first_position_count': first_position_count,
                    'first_position_pct': first_position_pct,
                    'usage_pct': (data['count'] / results[outcome]['count'] * 100)
                })
        
        stats_df = pd.DataFrame(stats_data)
        
        # Print detailed analysis
        print(f"\n{'='*80}")
        print(f"MODALITY RANKING ANALYSIS - Which Modality Contributes Most?")
        print(f"{'='*80}")
        print(f"\nPosition 1 = Primary contributor (most important)")
        print(f"Position 2 = Secondary contributor")
        print(f"Position 3+ = Lower priority contributors")
        
        outcome_names = {
            'TP': 'TRUE POSITIVES (Correct Violence Detection)',
            'TN': 'TRUE NEGATIVES (Correct Non-Violence)',
            'FP': 'FALSE POSITIVES (False Alarms)',
            'FN': 'FALSE NEGATIVES (Missed Violence)'
        }
        
        for outcome in ['TP', 'FN', 'FP', 'TN']:
            if outcome not in stats_df['outcome'].values:
                continue
            
            outcome_data = stats_df[stats_df['outcome'] == outcome].sort_values('first_position_pct', ascending=False)
            
            print(f"\n{'-'*80}")
            print(f"{outcome_names[outcome]}")
            print(f"{'-'*80}")
            
            print(f"\n{'Modality':<15} {'Used':<8} {'Avg Pos':<10} {'1st Place':<12} {'Impact'}")
            print(f"{'-'*80}")
            
            for _, row in outcome_data.iterrows():
                modality = row['modality']
                usage_pct = row['usage_pct']
                avg_pos = row['avg_position']
                first_pct = row['first_position_pct']
                first_count = row['first_position_count']
                
                # Determine impact level
                if first_pct >= 50 and usage_pct >= 50:
                    impact = "HIGH - Primary contributor"
                elif first_pct >= 30 or usage_pct >= 40:
                    impact = "MEDIUM - Significant role"
                elif first_pct >= 10 or usage_pct >= 20:
                    impact = "LOW - Supporting role"
                else:
                    impact = "💤 MINIMAL - Rarely primary"
                
                print(f"{modality:<15} {usage_pct:>5.1f}%   {avg_pos:>5.2f}      "
                      f"{first_count:>3}x ({first_pct:>4.1f}%)  {impact}")
        
        # Create visualizations
        self._create_modality_ranking_plots(stats_df, results, threshold)
    
    def _create_modality_ranking_plots(self, stats_df: pd.DataFrame, results: dict, threshold: float):
        """Create visualizations for modality ranking analysis."""
        
        # Get unique modalities
        all_modalities = sorted(stats_df['modality'].unique())
        outcome_order = ['TP', 'TN', 'FP', 'FN']
        outcome_labels = {
            'TP': f'True Positive\n(n={results.get("TP", {}).get("count", 0)})',
            'TN': f'True Negative\n(n={results.get("TN", {}).get("count", 0)})',
            'FP': f'False Positive\n(n={results.get("FP", {}).get("count", 0)})',
            'FN': f'False Negative\n(n={results.get("FN", {}).get("count", 0)})'
        }
        
        # Figure 1: Primary Contributor Heatmap (% times modality was ranked #1)
        primary_matrix = []
        avg_position_matrix = []
        labels_list = []
        
        for outcome in outcome_order:
            if outcome in stats_df['outcome'].values:
                row_primary = []
                row_avg_pos = []
                
                for modality in all_modalities:
                    mod_data = stats_df[(stats_df['outcome'] == outcome) & 
                                       (stats_df['modality'] == modality)]
                    if len(mod_data) > 0:
                        row_primary.append(mod_data['first_position_pct'].values[0])
                        row_avg_pos.append(mod_data['avg_position'].values[0])
                    else:
                        row_primary.append(0)
                        row_avg_pos.append(0)
                
                primary_matrix.append(row_primary)
                avg_position_matrix.append(row_avg_pos)
                labels_list.append(outcome_labels[outcome])
        
        # Plot 1: Primary Contributor Percentage
        fig1 = go.Figure(data=go.Heatmap(
            z=primary_matrix,
            x=all_modalities,
            y=labels_list,
            text=np.array(primary_matrix).round(1),
            texttemplate='%{text}%',
            textfont={"size": 11},
            colorscale='YlOrRd',
            colorbar=dict(title="% Times<br>Ranked #1")
        ))
        
        fig1.update_layout(
            title=f'🔥 Primary Contributor Analysis - Which Modality Ranked #1? (threshold={threshold})',
            xaxis_title='Modality',
            yaxis_title='Outcome Type',
            width=900,
            height=500,
            font=dict(size=12)
        )
        
        fig1.show()
        
        # Plot 2: Stacked bar chart showing position distribution
        self._plot_position_distribution(stats_df, results, threshold)
        
        # print(f"\nKey insights from modality ranking analysis:")
        # print(f"   Hotter colors in first heatmap = Modality frequently ranked #1 (primary contributor)")
        # print(f"   • Compare FN vs TP: Which modalities drop in ranking when violence is missed?")
        # print(f"   • Compare FP vs TN: Which modalities rank higher when causing false alarms?")
    
    def _plot_position_distribution(self, stats_df: pd.DataFrame, results: dict, threshold: float):
        """Plot detailed position distribution for key outcomes."""
        
        # Get position data for each outcome
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        outcome_info = [
            ('TP', 'True Positives (Correct Violence)', 0, 'green'),
            ('FN', 'False Negatives (Missed Violence)', 1, 'orange'),
            ('FP', 'False Positives (False Alarms)', 2, 'red'),
            ('TN', 'True Negatives (Correct Non-Violence)', 3, 'blue')
        ]
        
        for outcome, title, idx, color in outcome_info:
            ax = axes[idx]
            outcome_df = self.df[self.df['outcome'] == outcome]
            
            if len(outcome_df) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
                ax.set_title(title)
                continue
            
            # Count modality positions
            position_counts = defaultdict(Counter)
            for _, row in outcome_df.iterrows():
                for position, modality in enumerate(row['modalities_list'], start=1):
                    position_counts[position][modality] += 1
            
            # Prepare data for stacked bar
            modalities = sorted(stats_df[stats_df['outcome'] == outcome]['modality'].unique())
            positions = sorted(position_counts.keys())
            
            bottom = np.zeros(len(positions))
            
            for modality in modalities:
                values = [position_counts[pos][modality] for pos in positions]
                ax.bar(positions, values, bottom=bottom, label=modality, alpha=0.8)
                bottom += values
            
            ax.set_xlabel('Position (1=Primary, 2=Secondary, etc.)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax.set_title(f'{title}\n(n={len(outcome_df)})', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(alpha=0.3, axis='y')
            ax.set_xticks(positions)
        
        plt.tight_layout()
        plt.show()

    def generate_complete_evaluation_report(self, threshold: float = 0.5, save_dir: str = None):
        """
        Generate a comprehensive evaluation report with all metrics and visualizations.
        
        Args:
            threshold: Probability threshold for binary classification (default: 0.5)
            save_dir: Directory to save plots (optional)
        
        Returns:
            dict: Complete evaluation results containing all metrics
        """
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE EVALUATION REPORT - Threshold: {threshold:.2f}")
        print(f"{'='*80}\n")
        
        # Store all results
        report = {
            'threshold': threshold,
            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Binary Classification Metrics
        metrics = self.evaluate_with_threshold(threshold=threshold)
        report['binary_metrics'] = metrics
        
        # Confusion Matrix & Classification Report
        save_path_cm = f"{save_dir}/confusion_matrix.png" if save_dir else None
        self.plot_confusion_matrix(metrics, save_path=save_path_cm)
        
        save_path_cr = f"{save_dir}/classification_report.png" if save_dir else None
        self.plot_classification_report(metrics, save_path=save_path_cr)
        
        # ROC Curve Analysis
        save_path_roc = f"{save_dir}/roc_curve.png" if save_dir else None
        roc_results = self.plot_roc_curve_analysis(save_path=save_path_roc)
        report['roc_analysis'] = roc_results
        
        # Probability Calibration Metrics
        calib_results = self.calculate_probability_metrics()
        report['calibration_metrics'] = calib_results
        
        # Modality Combination Analysis
        modality_combo = self.analyze_modality_combinations(threshold=threshold)
        report['modality_combinations'] = modality_combo
        
        # Modality Performance by Outcome
        modality_results = self.analyze_modality_performance_by_outcome(threshold=threshold)
        report['modality_by_outcome'] = modality_results
        
        # Summary
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"AUC-ROC: {roc_results['auc_roc']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Brier Score: {calib_results['brier_score']:.4f}")
        print(f"Log Loss: {calib_results['log_loss']:.4f}")
        print(f"{'='*80}\n")
        
        return report