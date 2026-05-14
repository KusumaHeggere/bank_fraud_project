from evidently.report import Report
from evidently.metrics import DataDriftTable, DatasetDriftMetric
import pandas as pd
import joblib

def check_data_drift(reference_data, current_data):
    """Monitor for data drift"""
    report = Report(metrics=[DataDriftTable(), DatasetDriftMetric()])
    report.run(reference_data=reference_data, current_data=current_data)
    return report

def log_performance(y_true, y_pred, y_proba):
    """Log model performance metrics"""
    from sklearn.metrics import accuracy_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'timestamp': pd.Timestamp.now()
    }
    
    # Save to CSV for tracking
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv('models/performance_log.csv', mode='a', header=False)
    
    return metrics