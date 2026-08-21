from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

print("Testing /health endpoint...")
response = client.get('/health')
print(f"✅ Status: {response.status_code}")
print(f"   Response: {response.json()}")

print("\nTesting /generate endpoint...")
response = client.post('/generate', json={'email': 'Hi, can you help me?', 'num_suggestions': 1})
print(f"✅ Status: {response.status_code}")
result = response.json()
print(f"   Model loaded: {result.get('model_loaded')}")
if result.get('reply'):
    print(f"   Reply: {result.get('reply')[:100]}...")
else:
    print("   No reply")
