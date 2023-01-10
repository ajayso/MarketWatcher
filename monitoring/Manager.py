from ml_performance_monitoring.monitor import MLPerformanceMonitoring
from ml_performance_monitoring.psi import calculate_psi
import numpy as np
import pandas as pd
class MLLogger:
    def __init__(self):
        pass
    def register_Model(self,insert_key, model_name,metadata,f_columns,l_columns,label_type, model_version):
            self.ml_monitor = MLPerformanceMonitoring(
                insert_key=insert_key,  
                model_name=model_name,
                metadata=metadata,
                features_columns=list(f_columns),
                labels_columns=list(l_columns),
                label_type=label_type,
                model_version=model_version
            )
            self.features_columns= list(f_columns)
            self.labels_columns=list(l_columns)
            return(self.ml_monitor)

    def record_interface_data(self, X,Y):
        X_df = pd.DataFrame(
            X,
            columns=self.features_columns,
        )

        y_pred_df = pd.DataFrame(
            Y,
            columns=self.labels_columns,
        )
        print(X_df.shape)
        print(y_pred_df.shape)
        self.ml_monitor.record_inference_data(X_df, y_pred_df)
    
    def record_metrics(self,accuracy_score):
        metrics = {
            "Accuracy": accuracy_score
        }
        self.ml_monitor.record_metrics(metrics=metrics)
    
    def drift(self,X_train,X_test,y_train,y_pred):
        df_validation = np.transpose(X_test)
        df_training = np.transpose(X_train)
        top_feature_list = [0, 1, 2, 3]
        data_drift_metrics = {}
        psi_list = []
        for index, feature_name in enumerate(self.features_columns):
            # Assuming you have a validation and training set
            psi_t = calculate_psi(
                df_validation[index],
                df_training[index],
                buckettype="quantiles",
                buckets=10,
                axis=1,
            )
        self.ml_monitor.record_metrics(metrics={"data_drift": psi_t}, feature_name=feature_name)
        model_drift = calculate_psi(y_pred, y_train, buckettype="quantiles", buckets=10, axis=1)
        self.ml_monitor.record_metrics(metrics={"model_drift": model_drift})