# Test imports
print("Testing installations...\n")

try:
    import numpy as np
    print("NumPy:", np.__version__)
except:
    print("NumPy FAILED")

try:
    import pandas as pd
    print("Pandas:", pd.__version__)
except:
    print("Pandas FAILED")

try:
    import matplotlib
    print("Matplotlib:", matplotlib.__version__)
except:
    print("Matplotlib FAILED")

try:
    import nibabel as nib
    print("Nibabel:", nib.__version__)
except:
    print("Nibabel FAILED")

try:
    import nilearn
    print("Nilearn:", nilearn.__version__)
except:
    print("Nilearn FAILED")

try:
    import networkx as nx
    print("NetworkX:", nx.__version__)
except:
    print("NetworkX FAILED")

try:
    import scipy
    print("SciPy:", scipy.__version__)
except:
    print("SciPy FAILED")

try:
    from sklearn import __version__ as sklearn_version
    print("Scikit-learn:", sklearn_version)
except:
    print("Scikit-learn FAILED")

try:
    import statsmodels
    print("Statsmodels:", statsmodels.__version__)
except:
    print("Statsmodels FAILED")

try:
    import pingouin
    print("Pingouin:", pingouin.__version__)
except:
    print("Pingouin FAILED")

print("\nInstallation check complete!")