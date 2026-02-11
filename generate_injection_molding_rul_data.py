"""
Synthetic RUL Dataset Generator for Injection Molding Machine
Simulates degradation of key components over operational cycles
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

class InjectionMoldingSimulator:
    def __init__(self, num_machines=50):
        self.num_machines = num_machines
        
    def generate_machine_data(self, machine_id, cycles=1000):
        """Generate operational data for a single machine until failure"""
        
        # Random failure point (RUL starts from this and counts down)
        actual_failure_cycle = np.random.randint(800, 1200)
        
        # Initial conditions
        screw_wear = np.random.uniform(0.05, 0.15)  # Initial wear
        barrel_wear = np.random.uniform(0.03, 0.12)
        heater_degradation = np.random.uniform(0.02, 0.08)
        hydraulic_degradation = np.random.uniform(0.01, 0.05)
        
        # Degradation rates (random for each machine)
        screw_wear_rate = np.random.uniform(0.0008, 0.0015)
        barrel_wear_rate = np.random.uniform(0.0005, 0.001)
        heater_deg_rate = np.random.uniform(0.0003, 0.0008)
        hydraulic_deg_rate = np.random.uniform(0.0002, 0.0006)
        
        data = []
        
        for cycle in range(min(cycles, actual_failure_cycle)):
            # Calculate RUL (cycles remaining until failure)
            rul = actual_failure_cycle - cycle
            
            # Simulate component degradation (accelerates near failure)
            degradation_factor = 1 + (cycle / actual_failure_cycle) ** 2
            
            screw_wear += screw_wear_rate * degradation_factor
            barrel_wear += barrel_wear_rate * degradation_factor
            heater_degradation += heater_deg_rate * degradation_factor
            hydraulic_degradation += hydraulic_deg_rate * degradation_factor
            
            # Operating parameters affected by degradation
            # Injection pressure increases with wear
            injection_pressure = 80 + screw_wear * 100 + np.random.normal(0, 2)
            
            # Barrel temperature becomes less stable
            barrel_temp = 230 + heater_degradation * 50 + np.random.normal(0, 5 * heater_degradation)
            
            # Cycle time increases with degradation
            cycle_time = 45 + (screw_wear + barrel_wear) * 20 + np.random.normal(0, 1)
            
            # Clamping force becomes inconsistent
            clamping_force = 500 + hydraulic_degradation * 80 + np.random.normal(0, 10 * hydraulic_degradation)
            
            # Screw RPM decreases with wear
            screw_rpm = 100 - screw_wear * 30 + np.random.normal(0, 3)
            
            # Melt temperature affected by barrel wear and heater
            melt_temp = 240 + barrel_wear * 40 + heater_degradation * 30 + np.random.normal(0, 3)
            
            # Hydraulic pressure becomes erratic
            hydraulic_pressure = 150 + hydraulic_degradation * 60 + np.random.normal(0, 5 * hydraulic_degradation)
            
            # Power consumption increases with degradation
            power_consumption = 25 + (screw_wear + barrel_wear + heater_degradation) * 15 + np.random.normal(0, 2)
            
            # Vibration increases with mechanical wear
            vibration = 0.5 + (screw_wear + barrel_wear) * 3 + np.random.normal(0, 0.2)
            
            # Part quality degrades (defect rate increases)
            defect_rate = 0.01 + (screw_wear + barrel_wear + heater_degradation) * 0.05 + np.random.uniform(0, 0.02)
            
            # Operating conditions
            ambient_temp = np.random.normal(25, 5)
            material_viscosity = np.random.uniform(800, 1200)  # Pa·s
            shot_size = np.random.uniform(90, 110)  # grams
            
            data.append({
                'machine_id': machine_id,
                'cycle': cycle,
                'injection_pressure_bar': max(0, injection_pressure),
                'barrel_temp_celsius': max(0, barrel_temp),
                'cycle_time_seconds': max(0, cycle_time),
                'clamping_force_kN': max(0, clamping_force),
                'screw_rpm': max(0, screw_rpm),
                'melt_temp_celsius': max(0, melt_temp),
                'hydraulic_pressure_bar': max(0, hydraulic_pressure),
                'power_consumption_kW': max(0, power_consumption),
                'vibration_mm_s': max(0, vibration),
                'defect_rate': np.clip(defect_rate, 0, 1),
                'ambient_temp_celsius': ambient_temp,
                'material_viscosity_pas': material_viscosity,
                'shot_size_grams': shot_size,
                'screw_wear_index': np.clip(screw_wear, 0, 1),
                'barrel_wear_index': np.clip(barrel_wear, 0, 1),
                'heater_degradation_index': np.clip(heater_degradation, 0, 1),
                'hydraulic_degradation_index': np.clip(hydraulic_degradation, 0, 1),
                'RUL': rul  # Target variable
            })
        
        return pd.DataFrame(data)
    
    def generate_dataset(self):
        """Generate complete dataset for all machines"""
        all_data = []
        
        print(f"Generating data for {self.num_machines} machines...")
        for machine_id in range(1, self.num_machines + 1):
            machine_df = self.generate_machine_data(machine_id)
            all_data.append(machine_df)
            if machine_id % 10 == 0:
                print(f"  Completed {machine_id}/{self.num_machines} machines")
        
        dataset = pd.concat(all_data, ignore_index=True)
        return dataset

# Generate the dataset
print("=" * 60)
print("Injection Molding Machine RUL Dataset Generator")
print("=" * 60)

simulator = InjectionMoldingSimulator(num_machines=50)
dataset = simulator.generate_dataset()

# Save to CSV
output_file = os.path.join(os.path.dirname(__file__), 'data', 'injection_molding_rul_dataset.csv')
dataset.to_csv(output_file, index=False)

print(f"\n✓ Dataset generated successfully!")
print(f"  Total samples: {len(dataset)}")
print(f"  Machines: {dataset['machine_id'].nunique()}")
print(f"  Features: {len(dataset.columns) - 1}")  # Excluding RUL
print(f"  Saved to: {output_file}")

# Display statistics
print("\n" + "=" * 60)
print("Dataset Statistics:")
print("=" * 60)
print(dataset.describe())

print("\n" + "=" * 60)
print("RUL Distribution:")
print("=" * 60)
print(f"  Min RUL: {dataset['RUL'].min()}")
print(f"  Max RUL: {dataset['RUL'].max()}")
print(f"  Mean RUL: {dataset['RUL'].mean():.2f}")
print(f"  Median RUL: {dataset['RUL'].median():.2f}")

print("\n" + "=" * 60)
print("Sample Data (first 10 rows):")
print("=" * 60)
print(dataset.head(10))
