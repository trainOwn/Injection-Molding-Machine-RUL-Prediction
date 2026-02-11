"""
RUL Prediction Model Training Script for Injection Molding Machines
Trains multiple models and selects the best performer
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
import joblib
import warnings
warnings.filterwarnings('ignore')

class RULModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and prepare the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"  Loaded {len(self.df)} samples from {self.df['machine_id'].nunique()} machines")
        
        # Define feature columns (exclude machine_id, cycle, and RUL)
        self.feature_cols = [
            'injection_pressure_bar', 'barrel_temp_celsius', 'cycle_time_seconds',
            'clamping_force_kN', 'screw_rpm', 'melt_temp_celsius',
            'hydraulic_pressure_bar', 'power_consumption_kW', 'vibration_mm_s',
            'defect_rate', 'ambient_temp_celsius', 'material_viscosity_pas',
            'shot_size_grams', 'screw_wear_index', 'barrel_wear_index',
            'heater_degradation_index', 'hydraulic_degradation_index'
        ]
        
        self.target_col = 'RUL'
        
    def prepare_data(self, test_size=0.2, val_size=0.1):
        """Split and scale the data"""
        print("\nPreparing data...")
        
        X = self.df[self.feature_cols]
        y = self.df[self.target_col]
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = X_train_scaled
        self.X_val = X_val_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        print(f"  Train set: {len(X_train)} samples")
        print(f"  Validation set: {len(X_val)} samples")
        print(f"  Test set: {len(X_test)} samples")
        
    def train_models(self):
        """Train multiple regression models"""
        print("\n" + "="*60)
        print("Training Models")
        print("="*60)
        
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            )
        }
        
        if XGBOOST_AVAILABLE:
            models_to_train['XGBoost'] = XGBRegressor(
                n_estimators=100, max_depth=7, learning_rate=0.1, 
                random_state=42, n_jobs=-1
            )
        
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            
            # Train
            model.fit(self.X_train, self.y_train)
            
            # Predict on all sets
            y_train_pred = model.predict(self.X_train)
            y_val_pred = model.predict(self.X_val)
            y_test_pred = model.predict(self.X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            
            val_rmse = np.sqrt(mean_squared_error(self.y_val, y_val_pred))
            val_mae = mean_absolute_error(self.y_val, y_val_pred)
            val_r2 = r2_score(self.y_val, y_val_pred)
            
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'train_rmse': train_rmse,
                'train_mae': train_mae,
                'train_r2': train_r2,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'val_r2': val_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'predictions': y_test_pred
            }
            
            print(f"  Train RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}, R²: {train_r2:.4f}")
            print(f"  Val   RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}, R²: {val_r2:.4f}")
            print(f"  Test  RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R²: {test_r2:.4f}")
        
    def select_best_model(self):
        """Select the best model based on validation RMSE"""
        print("\n" + "="*60)
        print("Model Comparison")
        print("="*60)
        
        # Create comparison dataframe
        comparison = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Val RMSE': [r['val_rmse'] for r in self.results.values()],
            'Val MAE': [r['val_mae'] for r in self.results.values()],
            'Val R²': [r['val_r2'] for r in self.results.values()],
            'Test RMSE': [r['test_rmse'] for r in self.results.values()],
            'Test MAE': [r['test_mae'] for r in self.results.values()],
            'Test R²': [r['test_r2'] for r in self.results.values()]
        })
        
        comparison = comparison.sort_values('Val RMSE')
        print("\n", comparison.to_string(index=False))
        
        # Select best model
        best_model_name = comparison.iloc[0]['Model']
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        print(f"\n✓ Best Model: {best_model_name}")
        print(f"  Validation RMSE: {self.results[best_model_name]['val_rmse']:.2f}")
        print(f"  Test RMSE: {self.results[best_model_name]['test_rmse']:.2f}")
        print(f"  Test R²: {self.results[best_model_name]['test_r2']:.4f}")
        
        return comparison
    
    def feature_importance(self):
        """Display feature importance for the best model"""
        if self.best_model_name not in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
            print("\nFeature importance not available for this model type")
            return
        
        print("\n" + "="*60)
        print("Feature Importance (Top 10)")
        print("="*60)
        
        importance = self.best_model.feature_importances_
        feature_imp = pd.DataFrame({
            'Feature': self.feature_cols,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print(feature_imp.head(10).to_string(index=False))
        
        return feature_imp
    
    def plot_results(self):
        """Create visualization plots"""
        print("\nGenerating plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Predicted (Best Model)
        best_pred = self.results[self.best_model_name]['predictions']
        axes[0, 0].scatter(self.y_test, best_pred, alpha=0.5, s=10)
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()], 
                        [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual RUL (cycles)', fontsize=12)
        axes[0, 0].set_ylabel('Predicted RUL (cycles)', fontsize=12)
        axes[0, 0].set_title(f'Actual vs Predicted RUL - {self.best_model_name}', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residual Plot
        residuals = self.y_test.values - best_pred
        axes[0, 1].scatter(best_pred, residuals, alpha=0.5, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted RUL (cycles)', fontsize=12)
        axes[0, 1].set_ylabel('Residuals', fontsize=12)
        axes[0, 1].set_title('Residual Plot', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Model Comparison (RMSE)
        model_names = list(self.results.keys())
        test_rmse = [self.results[m]['test_rmse'] for m in model_names]
        
        colors = ['green' if m == self.best_model_name else 'skyblue' for m in model_names]
        axes[1, 0].barh(model_names, test_rmse, color=colors)
        axes[1, 0].set_xlabel('Test RMSE', fontsize=12)
        axes[1, 0].set_title('Model Comparison - RMSE', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # 4. Prediction Error Distribution
        axes[1, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Prediction Error (cycles)', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_title('Prediction Error Distribution', fontsize=14)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(__file__), 'results', 'rul_model_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved plot to: {plot_path}")
        
        # Feature importance plot (if applicable)
        if self.best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
            feature_imp = self.feature_importance()
            
            plt.figure(figsize=(10, 8))
            top_features = feature_imp.head(15)
            plt.barh(range(len(top_features)), top_features['Importance'])
            plt.yticks(range(len(top_features)), top_features['Feature'])
            plt.xlabel('Importance', fontsize=12)
            plt.title(f'Feature Importance - {self.best_model_name}', fontsize=14)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            fi_plot_path = os.path.join(os.path.dirname(__file__), 'results', 'feature_importance.png')
            plt.savefig(fi_plot_path, dpi=300, bbox_inches='tight')
            print(f"  Saved feature importance plot to: {fi_plot_path}")
    
    def save_model(self):
        """Save the best model and scaler"""
        print("\nSaving model artifacts...")
        
        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(base_dir, 'models', 'best_rul_model.pkl')
        scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
        
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"  ✓ Model saved to: {model_path}")
        print(f"  ✓ Scaler saved to: {scaler_path}")
        
        # Save feature names
        feature_names_path = os.path.join(base_dir, 'models', 'feature_names.txt')
        with open(feature_names_path, 'w') as f:
            f.write('\n'.join(self.feature_cols))
        print(f"  ✓ Feature names saved to: {feature_names_path}")
    
    def run_pipeline(self):
        """Execute the complete training pipeline"""
        self.load_data()
        self.prepare_data()
        self.train_models()
        comparison = self.select_best_model()
        self.plot_results()
        self.save_model()
        
        return comparison

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("RUL Prediction Model Training Pipeline")
    print("="*60)
    
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'injection_molding_rul_dataset.csv')
    trainer = RULModelTrainer(data_path)
    comparison = trainer.run_pipeline()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nFiles created:")
    print("  1. best_rul_model.pkl - Trained model")
    print("  2. scaler.pkl - Feature scaler")
    print("  3. feature_names.txt - Feature list")
    print("  4. rul_model_results.png - Visualization")
    print("  5. feature_importance.png - Feature importance plot")
