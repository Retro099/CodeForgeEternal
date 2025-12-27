#!/usr/bin/env python3
"""
CodeForgeEternal - Hello World Test Script
Run with: python src/hello_world.py
Verifies: Python env, requirements.txt, basic imports
"""

import sys
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

print("🚀 CodeForgeEternal Environment Test")
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")

# Quick ML test
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(f"✅ Loaded Iris dataset: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\n🎉 Environment ready for ML projects!")