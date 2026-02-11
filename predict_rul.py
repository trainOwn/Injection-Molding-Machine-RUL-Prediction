"""
RUL Prediction Inference Script
Use the trained model to predict Remaining Useful Life
"""

import os
import numpy as np
import pandas as pd
import joblib
import argparse

class RULPredictor:
    def __init__(self, model_path=None,
                 scaler_path=None,
                 feature_names_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = model_path or os.path.join(base_dir, 'models', 'best_rul_model.pkl')
        scaler_path = scaler_path or os.path.join(base_dir, 'models', 'scaler.pkl')
        feature_names_path = feature_names_path or os.path.join(base_dir, 'models', 'feature_names.txt')
        """Initialize the predictor by loading model artifacts"""
        print("Loading model artifacts...")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        with open(feature_names_path, 'r') as f:
            self.feature_names = [line.strip() for line in f]
        
        print(f"✓ Model loaded: {type(self.model).__name__}")
        print(f"✓ Features required: {len(self.feature_names)}")
    
    def predict_single(self, features_dict):
        """Predict RUL for a single sample"""
        # Convert dict to DataFrame to maintain feature order
        features_df = pd.DataFrame([features_dict])
        
        # Ensure all required features are present
        missing = set(self.feature_names) - set(features_df.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        # Select and order features correctly
        features_df = features_df[self.feature_names]
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Predict
        rul = self.model.predict(features_scaled)[0]
        
        return max(0, rul)  # RUL cannot be negative
    
    def predict_batch(self, csv_path):
        """Predict RUL for multiple samples from CSV"""
        print(f"\nLoading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Ensure all required features are present
        missing = set(self.feature_names) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        # Select and order features correctly
        features_df = df[self.feature_names]
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Predict
        predictions = self.model.predict(features_scaled)
        predictions = np.maximum(0, predictions)  # RUL cannot be negative
        
        # Add predictions to dataframe
        result_df = df.copy()
        result_df['predicted_RUL'] = predictions
        
        if 'RUL' in df.columns:
            result_df['error'] = result_df['RUL'] - result_df['predicted_RUL']
            result_df['abs_error'] = np.abs(result_df['error'])
        
        return result_df
    
    def maintenance_recommendation(self, rul, threshold_critical=50, threshold_warning=150):
        """Provide maintenance recommendations based on RUL"""
        if rul <= threshold_critical:
            status = "🔴 CRITICAL"
            recommendation = "Immediate maintenance required! Schedule downtime within 24-48 hours."
        elif rul <= threshold_warning:
            status = "🟡 WARNING"
            recommendation = "Maintenance should be scheduled within the next week."
        else:
            status = "🟢 HEALTHY"
            recommendation = "Machine is operating normally. Continue routine monitoring."
        
        return status, recommendation

def demo_prediction():
    """Demonstrate prediction with example data"""
    print("="*70)
    print("RUL Prediction Demo")
    print("="*70)
    
    predictor = RULPredictor()
    
    # Example 1: Healthy machine
    print("\n" + "-"*70)
    print("Example 1: Healthy Machine (Early Life)")
    print("-"*70)
    
    healthy_machine = {
        'injection_pressure_bar': 82.5,
        'barrel_temp_celsius': 232.0,
        'cycle_time_seconds': 46.2,
        'clamping_force_kN': 508.0,
        'screw_rpm': 98.5,
        'melt_temp_celsius': 242.0,
        'hydraulic_pressure_bar': 153.0,
        'power_consumption_kW': 26.5,
        'vibration_mm_s': 0.6,
        'defect_rate': 0.015,
        'ambient_temp_celsius': 24.0,
        'material_viscosity_pas': 1000.0,
        'shot_size_grams': 100.0,
        'screw_wear_index': 0.08,
        'barrel_wear_index': 0.06,
        'heater_degradation_index': 0.04,
        'hydraulic_degradation_index': 0.03
    }
    
    rul = predictor.predict_single(healthy_machine)
    status, rec = predictor.maintenance_recommendation(rul)
    
    print(f"\nPredicted RUL: {rul:.0f} cycles")
    print(f"Status: {status}")
    print(f"Recommendation: {rec}")
    
    # Example 2: Degraded machine
    print("\n" + "-"*70)
    print("Example 2: Degraded Machine (Near End of Life)")
    print("-"*70)
    
    degraded_machine = {
        'injection_pressure_bar': 105.0,
        'barrel_temp_celsius': 245.0,
        'cycle_time_seconds': 58.5,
        'clamping_force_kN': 560.0,
        'screw_rpm': 78.0,
        'melt_temp_celsius': 268.0,
        'hydraulic_pressure_bar': 185.0,
        'power_consumption_kW': 38.0,
        'vibration_mm_s': 2.8,
        'defect_rate': 0.085,
        'ambient_temp_celsius': 26.0,
        'material_viscosity_pas': 950.0,
        'shot_size_grams': 98.0,
        'screw_wear_index': 0.72,
        'barrel_wear_index': 0.68,
        'heater_degradation_index': 0.55,
        'hydraulic_degradation_index': 0.48
    }
    
    rul = predictor.predict_single(degraded_machine)
    status, rec = predictor.maintenance_recommendation(rul)
    
    print(f"\nPredicted RUL: {rul:.0f} cycles")
    print(f"Status: {status}")
    print(f"Recommendation: {rec}")
    
    # Example 3: Warning state machine
    print("\n" + "-"*70)
    print("Example 3: Machine in Warning State (Mid-Life)")
    print("-"*70)
    
    warning_machine = {
        'injection_pressure_bar': 92.0,
        'barrel_temp_celsius': 238.0,
        'cycle_time_seconds': 51.0,
        'clamping_force_kN': 535.0,
        'screw_rpm': 88.0,
        'melt_temp_celsius': 254.0,
        'hydraulic_pressure_bar': 168.0,
        'power_consumption_kW': 32.0,
        'vibration_mm_s': 1.5,
        'defect_rate': 0.045,
        'ambient_temp_celsius': 25.0,
        'material_viscosity_pas': 980.0,
        'shot_size_grams': 99.0,
        'screw_wear_index': 0.42,
        'barrel_wear_index': 0.38,
        'heater_degradation_index': 0.32,
        'hydraulic_degradation_index': 0.28
    }
    
    rul = predictor.predict_single(warning_machine)
    status, rec = predictor.maintenance_recommendation(rul)
    
    print(f"\nPredicted RUL: {rul:.0f} cycles")
    print(f"Status: {status}")
    print(f"Recommendation: {rec}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict RUL for injection molding machines')
    parser.add_argument('--demo', action='store_true', help='Run demo predictions')
    parser.add_argument('--csv', type=str, help='Path to CSV file for batch prediction')
    parser.add_argument('--output', type=str, help='Output path for batch predictions')
    
    args = parser.parse_args()
    
    if args.demo or (not args.csv):
        demo_prediction()
    
    if args.csv:
        predictor = RULPredictor()
        results = predictor.predict_batch(args.csv)
        
        output_path = args.output or 'predictions_output.csv'
        results.to_csv(output_path, index=False)
        
        print(f"\n✓ Batch predictions saved to: {output_path}")
        
        if 'RUL' in results.columns:
            print(f"\nPrediction Statistics:")
            print(f"  Mean Absolute Error: {results['abs_error'].mean():.2f} cycles")
            print(f"  RMSE: {np.sqrt((results['error']**2).mean()):.2f} cycles")
