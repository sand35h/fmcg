import requests
import sys

try:
    # Test Model Info
    r = requests.get('http://localhost:8000/model-info')
    print("Model Info Status:", r.status_code)
    print("Model Info Data:", r.json())
    
    # Test Feature Importance
    r = requests.get('http://localhost:8000/feature-importance')
    print("\nFeature Importance Status:", r.status_code)
    data = r.json()
    print(f"Feature Importance Count: {len(data)}")
    print("First Feature:", data[0] if data else "None")
    
except Exception as e:
    print(f"Error: {e}")
