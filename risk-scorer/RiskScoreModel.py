from dubbo.client import DubboClient, ZkRegister
import onnxruntime as ort
import numpy as np
from env_loader import base

# @TODO get risk model
sess = ort.InferenceSession('risk_model.onnx', providers=['CPUExecutionProvider'])

class RiskScorerService:
    def score(self, scene, features):
        """Calculate risk score based on input features"""
        try:
            # Prepare input data
            x = np.array([features['values']], dtype=np.float32)
            
            # Run inference
            result = sess.run(None, {'input': x})[0]
            prob = np.array(result)[0][0][1].item()
            
            # Convert probability to score (800-500 range)
            score = int(800 - prob * 300)
            return score
        except Exception as e:
            # Fallback score calculation
            debt = features.get('external_debt_ratio', 0)
            age = features.get('age', 30)
            return int(700 - debt * 120 - max(0, 25 - age) * 2)

def start_risk_score_model():

    # Dubbo service configuration
    interface = base.service_package_name
    
    # Zookeeper registry
    zk = ZkRegister(base.zk_port)
    
    # Create Dubbo client
    client = DubboClient(interface, zk_register=zk)
    
    # Register service methods
    client.register_method('score', RiskScorerService().score)
