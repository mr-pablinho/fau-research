#!/usr/bin/env python3
"""
Quick analysis of DREAM results data
"""

import pandas as pd
import numpy as np

def analyze_dream_csv(filename="DREAM_GWM_test.csv"):
    """Quick analysis of DREAM results"""
    
    print(f"📊 Analyzing: {filename}")
    print("="*50)
    
    # Load and display basic info
    df = pd.read_csv(filename)
    print(f"📋 Shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    print("\n📊 First few rows:")
    print(df.head())
    
    print("\n📊 Statistical Summary:")
    print(df.describe())
    
    # Find best iteration
    best_idx = df['like1'].idxmax()
    print(f"\n🏆 Best Result (Row {best_idx}):")
    print(df.iloc[best_idx])
    
    # Parameter evolution
    param_cols = [col for col in df.columns if col.startswith('par')]
    print(f"\n📈 Parameter Evolution:")
    for param in param_cols:
        values = df[param].values
        print(f"   {param}: {values}")
    
    print(f"\n🎯 Objective Evolution:")
    print(f"   like1: {df['like1'].values}")
    
    return df

if __name__ == "__main__":
    analyze_dream_csv()
