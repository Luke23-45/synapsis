#!/usr/bin/env python3
"""
Comprehensive analysis of the exported episode CSV.
Analyzes every single column and data aspect.
"""
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats

def analyze_csv(csv_path):
    """Comprehensive CSV analysis."""
    print(f"Reading {csv_path}...")
    
    # Read with pandas for easier analysis
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*70)
    print("COMPREHENSIVE CSV ANALYSIS")
    print("="*70)
    
    # 1. Basic Info
    print("\n[1] BASIC INFORMATION")
    print(f"  Total rows: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Episode ID: {df['episode_id'].iloc[0]}")
    print(f"  Step range: {df['step'].min()} to {df['step'].max()}")
    
    # 2. Column Types
    print("\n[2] COLUMN TYPES")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"  Numeric columns: {len(numeric_cols)}")
    print(f"  String columns: {len(string_cols)}")
    if string_cols:
        print(f"  String columns: {string_cols}")
    
    # 3. Missing Values
    print("\n[3] MISSING VALUES")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"  ⚠️  Missing values found:")
        for col, count in missing[missing > 0].items():
            print(f"    {col}: {count} ({count/len(df)*100:.2f}%)")
    else:
        print("  ✅ No missing values")
    
    # 4. Numeric Column Statistics
    print("\n[4] NUMERIC COLUMN STATISTICS")
    print(f"  Analyzing {len(numeric_cols)} numeric columns...")
    
    issues = []
    for col in numeric_cols:
        if col in ['step', 'episode_id']:
            continue
            
        values = df[col].dropna()
        
        # Basic stats
        mean = values.mean()
        std = values.std()
        min_val = values.min()
        max_val = values.max()
        
        # Check for NaN/Inf
        if np.isnan(values).any():
            issues.append(f"{col}: Contains NaN values")
        if np.isinf(values).any():
            issues.append(f"{col}: Contains Inf values")
        
        # Check for zero variance
        if std < 1e-12:
            issues.append(f"{col}: Zero variance (constant value: {mean})")
        
        # Check for outliers (using IQR method)
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        outliers = values[(values < lower_bound) | (values > upper_bound)]
        if len(outliers) > 0:
            issues.append(f"{col}: {len(outliers)} outliers detected")
        
        # Print summary for key columns
        if 'action' in col or 'joint' in col or 'ee_' in col or 'delta' in col or 'target' in col:
            print(f"\n  {col}:")
            print(f"    Mean: {mean:.6f}")
            print(f"    Std:  {std:.6f}")
            print(f"    Min:  {min_val:.6f}")
            print(f"    Max:  {max_val:.6f}")
            print(f"    Range: {max_val - min_val:.6f}")
    
    # 5. Action Analysis
    print("\n[5] ACTION ANALYSIS")
    action_cols = [col for col in numeric_cols if 'action' in col and col != 'action_gripper']
    if action_cols:
        print(f"  Found {len(action_cols)} action columns")
        action_data = df[action_cols]
        print(f"  Action norm statistics:")
        action_norms = np.linalg.norm(action_data.values, axis=1)
        print(f"    Mean norm: {action_norms.mean():.6f}")
        print(f"    Max norm: {action_norms.max():.6f}")
        print(f"    Min norm: {action_norms.min():.6f}")
    
    # 6. Joint Angle Analysis
    print("\n[6] JOINT ANGLE ANALYSIS")
    joint_cols = [col for col in numeric_cols if 'joint' in col]
    if joint_cols:
        print(f"  Found {len(joint_cols)} joint columns")
        joint_data = df[joint_cols]
        for i, col in enumerate(joint_cols):
            print(f"  {col}:")
            print(f"    Range: [{joint_data[col].min():.4f}, {joint_data[col].max():.4f}]")
            print(f"    Mean: {joint_data[col].mean():.4f}")
    
    # 7. End-Effector Pose Analysis
    print("\n[7] END-EFFECTOR POSE ANALYSIS")
    ee_cols = [col for col in numeric_cols if 'ee_' in col and col != 'ee_vel']
    if ee_cols:
        print(f"  Found {len(ee_cols)} EE columns")
        # Position
        pos_cols = [col for col in ee_cols if 'x' in col or 'y' in col or 'z' in col]
        if pos_cols:
            pos_data = df[pos_cols]
            print(f"  Position statistics:")
            print(f"    X: [{pos_data['ee_x'].min():.4f}, {pos_data['ee_x'].max():.4f}]")
            print(f"    Y: [{pos_data['ee_y'].min():.4f}, {pos_data['ee_y'].max():.4f}]")
            print(f"    Z: [{pos_data['ee_z'].min():.4f}, {pos_data['ee_z'].max():.4f}]")
        
        # Quaternion
        quat_cols = [col for col in ee_cols if 'qx' in col or 'qy' in col or 'qz' in col or 'qw' in col]
        if quat_cols:
            quat_data = df[quat_cols]
            print(f"  Quaternion statistics:")
            for col in quat_cols:
                print(f"    {col}: [{quat_data[col].min():.4f}, {quat_data[col].max():.4f}]")
            # Check quaternion normalization
            quat_norms = np.linalg.norm(quat_data.values, axis=1)
            print(f"  Quaternion norms:")
            print(f"    Mean: {quat_norms.mean():.6f}")
            print(f"    Std:  {quat_norms.std():.6f}")
            if abs(quat_norms.mean() - 1.0) > 0.1:
                issues.append("Quaternions not normalized (mean norm != 1.0)")
    
    # 8. Delta Analysis
    print("\n[8] DELTA POSE ANALYSIS")
    delta_cols = [col for col in numeric_cols if 'delta' in col]
    if delta_cols:
        print(f"  Found {len(delta_cols)} delta columns")
        delta_data = df[delta_cols]
        print(f"  Delta position statistics:")
        if 'delta_x' in delta_data.columns:
            print(f"    X: [{delta_data['delta_x'].min():.6f}, {delta_data['delta_x'].max():.6f}]")
        if 'delta_y' in delta_data.columns:
            print(f"    Y: [{delta_data['delta_y'].min():.6f}, {delta_data['delta_y'].max():.6f}]")
        if 'delta_z' in delta_data.columns:
            print(f"    Z: [{delta_data['delta_z'].min():.6f}, {delta_data['delta_z'].max():.6f}]")
    
    # 9. Target Pose Analysis
    print("\n[9] TARGET POSE ANALYSIS")
    target_cols = [col for col in numeric_cols if 'target' in col]
    if target_cols:
        print(f"  Found {len(target_cols)} target columns")
        target_data = df[target_cols]
        print(f"  Target position statistics:")
        if 'target_x' in target_data.columns:
            print(f"    X: [{target_data['target_x'].min():.4f}, {target_data['target_x'].max():.4f}]")
        if 'target_y' in target_data.columns:
            print(f"    Y: [{target_data['target_y'].min():.4f}, {target_data['target_y'].max():.4f}]")
        if 'target_z' in target_data.columns:
            print(f"    Z: [{target_data['target_z'].min():.4f}, {target_data['target_z'].max():.4f}]")
    
    # 10. Gripper Analysis
    print("\n[10] GRIPPER ANALYSIS")
    if 'action_gripper' in df.columns:
        gripper = df['action_gripper']
        print(f"  Gripper values:")
        print(f"    Min: {gripper.min():.4f}")
        print(f"    Max: {gripper.max():.4f}")
        print(f"    Mean: {gripper.mean():.4f}")
        print(f"    Unique values: {gripper.nunique()}")
        print(f"    Value distribution:")
        print(gripper.value_counts())
    
    # 11. Object and Goal Analysis
    print("\n[11] OBJECT & GOAL ANALYSIS")
    obj_cols = [col for col in numeric_cols if 'obj_' in col]
    goal_cols = [col for col in numeric_cols if 'goal_' in col]
    if obj_cols:
        print(f"  Object position:")
        for col in obj_cols:
            print(f"    {col}: {df[col].iloc[0]:.4f} (assumed constant)")
    if goal_cols:
        print(f"  Goal position:")
        for col in goal_cols:
            print(f"    {col}: {df[col].iloc[0]:.4f} (assumed constant)")
    
    # 12. Expert State Analysis
    print("\n[12] EXPERT STATE ANALYSIS")
    if 'expert_phase' in df.columns:
        print(f"  Phase distribution:")
        print(df['expert_phase'].value_counts())
    if 'gripper_intent' in df.columns:
        print(f"  Gripper intent:")
        print(df['gripper_intent'].value_counts())
    if 'state_str' in df.columns:
        print(f"  State string distribution:")
        print(df['state_str'].value_counts())
    
    # 13. Temporal Consistency
    print("\n[13] TEMPORAL CONSISTENCY")
    # Check step sequence
    steps = df['step'].values
    if not np.array_equal(steps, np.arange(len(steps))):
        issues.append("Step sequence is not sequential")
    else:
        print("  ✅ Step sequence is sequential")
    
    # Check EE pose smoothness
    if 'ee_x' in df.columns and 'ee_y' in df.columns and 'ee_z' in df.columns:
        ee_pos = df[['ee_x', 'ee_y', 'ee_z']].values
        ee_vel = np.diff(ee_pos, axis=0)
        ee_vel_mag = np.linalg.norm(ee_vel, axis=1)
        print(f"  EE velocity statistics:")
        print(f"    Mean: {ee_vel_mag.mean():.6f}")
        print(f"    Max: {ee_vel_mag.max():.6f}")
        print(f"    Std: {ee_vel_mag.std():.6f}")
        if ee_vel_mag.max() > 0.5:
            issues.append(f"High EE velocity detected: {ee_vel_mag.max():.4f}")
    
    # 14. Issues Summary
    print("\n" + "="*70)
    print("ISSUES SUMMARY")
    print("="*70)
    if issues:
        print(f"  ⚠️  Total issues: {len(issues)}")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
    else:
        print("  ✅ No issues detected. Data looks healthy!")
    
    print("\n" + "="*70)
    print("ANYSIS COMPLETE")
    print("="*70)

def main():
    csv_path = Path("output_data/exports/episode_0_ep_000000/telemetry.csv")
    analyze_csv(csv_path)

if __name__ == "__main__":
    main()
