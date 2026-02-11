# RUL Dataset Documentation

**Injection Molding Machine Remaining Useful Life Prediction**

## 1. Dataset Overview

This synthetic dataset simulates operational data from 50 injection molding machines over their lifecycle until failure. The dataset is designed for predictive maintenance applications, specifically for predicting the Remaining Useful Life (RUL) of machines based on sensor readings and operational parameters.

### 1.1 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 47,211 |
| Number of Machines | 50 |
| Input Features | 17 |
| Target Variable | RUL (Remaining Useful Life) |
| RUL Range | 1 - 1,197 cycles |
| Mean RUL | 513.88 cycles |

## 2. Input Features

The dataset contains 17 input features organized into four categories: Process Parameters, Condition Indicators, Environmental Factors, and Component Degradation Indices.

### 2.1 Process Parameters

These features represent the core operational parameters of the injection molding process:

| Feature Name | Unit | Description |
|--------------|------|-------------|
| `injection_pressure_bar` | bar | Pressure applied during injection phase. Increases with screw wear as more force is needed to push material. |
| `barrel_temp_celsius` | °C | Barrel temperature where plastic is melted. Becomes less stable with heater degradation. |
| `cycle_time_seconds` | seconds | Total time for one complete molding cycle. Increases significantly with component wear (most important feature with 49.98% importance). |
| `clamping_force_kN` | kN | Force holding mold halves together. Becomes inconsistent with hydraulic system degradation. |
| `screw_rpm` | RPM | Screw rotation speed for plasticizing material. Decreases with screw wear. |
| `melt_temp_celsius` | °C | Temperature of molten plastic. Affected by barrel wear and heater performance. |
| `hydraulic_pressure_bar` | bar | Hydraulic system pressure. Becomes erratic with hydraulic system degradation. |
| `power_consumption_kW` | kW | Electrical power consumption. Increases with overall degradation as machine works harder. |

### 2.2 Condition Indicators

Features that directly indicate machine health and performance:

| Feature Name | Unit | Description |
|--------------|------|-------------|
| `vibration_mm_s` | mm/s | Machine vibration level. Increases with mechanical wear (screw and barrel). 4th most important feature. |
| `defect_rate` | ratio | Proportion of defective parts produced. Increases dramatically as machine degrades. 2nd most important feature with 26.74% importance. |

### 2.3 Environmental Factors

External conditions that may affect machine operation:

| Feature Name | Unit | Description |
|--------------|------|-------------|
| `ambient_temp_celsius` | °C | Factory floor temperature. Varies naturally and affects cooling rates. |
| `material_viscosity_pas` | Pa·s | Plastic material viscosity. Varies by batch and affects processing characteristics. |
| `shot_size_grams` | grams | Amount of plastic injected per cycle. Varies based on part being produced. |

### 2.4 Component Degradation Indices

Normalized indices (0-1 scale) representing the degradation state of key components:

| Feature Name | Description |
|--------------|-------------|
| `screw_wear_index` | Screw component wear (0=new, 1=completely worn). Affects injection pressure and RPM. 7th most important feature. |
| `barrel_wear_index` | Barrel component wear (0=new, 1=completely worn). Affects melt temperature. 6th most important feature. |
| `heater_degradation_index` | Heating element degradation (0=new, 1=failed). Causes temperature instability. 5th most important feature. |
| `hydraulic_degradation_index` | Hydraulic system degradation (0=new, 1=failed). Affects clamping force and hydraulic pressure. 3rd most important feature. |

## 3. Target Variable (Output)

| Variable Name | Unit | Description |
|---------------|------|-------------|
| `RUL` | cycles | Remaining Useful Life - the number of operational cycles remaining before machine failure. This is the prediction target. |

**Key characteristics of RUL:**

- Continuous numeric value representing cycles until failure
- Decreases monotonically from machine startup to failure
- Range: 1 to 1,197 cycles in this dataset
- Critical for predictive maintenance scheduling
- Enables proactive intervention before catastrophic failure

## 4. Degradation Model

The synthetic data is generated using a physics-informed degradation model that simulates realistic wear patterns:

### 4.1 Component Wear Progression

Each component (screw, barrel, heater, hydraulic) has:

- Initial wear level (randomized per machine)
- Base degradation rate (randomized per machine)
- Accelerated degradation factor (quadratic increase near failure)

### 4.2 Cascading Effects

Component degradation creates realistic cascading effects on operational parameters:

| Component Wear | Observable Effects |
|----------------|-------------------|
| Screw Wear | ↑ Injection pressure (harder to push material), ↓ Screw RPM, ↑ Vibration, ↑ Defect rate |
| Barrel Wear | ↑ Melt temperature, ↑ Cycle time, ↑ Vibration, ↑ Defect rate |
| Heater Degradation | Unstable barrel/melt temperatures, ↑ Defect rate, ↑ Power consumption |
| Hydraulic Degradation | Erratic clamping force and hydraulic pressure, ↑ Defect rate |

## 5. Feature Importance Analysis

Based on the trained Random Forest model, the following features are most predictive of RUL:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `cycle_time_seconds` | 49.98% |
| 2 | `defect_rate` | 26.74% |
| 3 | `hydraulic_degradation_index` | 5.30% |
| 4 | `vibration_mm_s` | 4.35% |
| 5 | `heater_degradation_index` | 3.42% |

**Key Insights:**

- Cycle time and defect rate together account for 76.72% of predictive power
- These are directly observable metrics that don't require specialized sensors
- Component degradation indices provide additional 13% of predictive power
- Environmental factors have minimal direct impact on RUL prediction

## 6. Practical Applications

### 6.1 Predictive Maintenance

The RUL prediction enables:

- Scheduled maintenance windows before failure occurs
- Optimized spare parts inventory management
- Reduced unplanned downtime and production losses
- Extended equipment lifespan through timely interventions

### 6.2 Maintenance Thresholds

| Status | RUL Range | Action |
|--------|-----------|--------|
| HEALTHY | > 150 cycles | Routine monitoring |
| WARNING | 50-150 cycles | Schedule within week |
| CRITICAL | < 50 cycles | Immediate (24-48 hrs) |

## 7. Data Quality Considerations

### 7.1 Synthetic Data Advantages

- Complete lifecycle data (from new to failure)
- Controlled degradation patterns
- No missing values or sensor failures
- Known ground truth for validation

### 7.2 Limitations

- Simplified physics model may not capture all real-world complexities
- No sudden failures or catastrophic events simulated
- All degradation follows smooth, predictable patterns
- Real machines may exhibit different failure modes not represented

### 7.3 Recommendations for Real-World Deployment

- Collect real failure data for model fine-tuning
- Implement anomaly detection for unexpected failure modes
- Use transfer learning to adapt to specific machine types
- Continuously update model with new operational data
- Validate predictions against actual maintenance records
