import sys
sys.path.insert(0, 'src')

from config import MODEL_CHECKPOINT_DIR
from inference import EmailReplyGenerator

try:
    print(f"Loading model from: {MODEL_CHECKPOINT_DIR}")
    print(f"Model path exists: {MODEL_CHECKPOINT_DIR.exists()}")
    
    generator = EmailReplyGenerator(model_path=MODEL_CHECKPOINT_DIR)
    print("✅ Generator initialized successfully!")
    
    test_reply = generator.generate("Hi, can you help me?")
    print(f"✅ Sample reply: {test_reply}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
