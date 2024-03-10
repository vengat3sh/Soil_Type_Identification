import tensorflow as tf
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

print("="*50)
print("All imports successful!")
print(f"TensorFlow version: {tf.__version__}")
print(f"Streamlit version: {st.__version__}")
print("="*50)

# Test model loading
try:
    model = tf.keras.models.load_model("trained_soil_model.keras")
    print("✅ Model loaded successfully!")
    
    # Get class names from model
    class_names = ['Alluvial_Soil', 'Arid_Soil', 'Black_Soil', 
                   'Laterite_Soil', 'Mountain_Soil', 'Red_Soil', 'Yellow_Soil']
    print(f"✅ Class names: {class_names}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")

print("="*50)
print("Environment is ready!")