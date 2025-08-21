from dubbo.codec.hessian2 import DubboCodec
from dubbo.client import DubboClient
from dubbo.common.constants import DubboVersion
from dubbo.registry.zookeeper import ZookeeperRegistry
from dubbo.server import DubboServer
import onnxruntime as ort
import numpy as np

# @TODO train risk model
sess = ort.InferenceSession('risk_model.onnx', providers=['CPUExecutionProvider'])

class RiskScorerService:
    def score(self, scene, features):
        """Calculate risk score based on input features"""
        try:
            # Prepare input data
            x = np.array([features['values']], dtype=np.float32)
            
            # Run inference
            prob = sess.run(None, {'input': x})[0][0][1].item()
            
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
    service = RiskScorerService()
    interface = 'com.metawebthree.common.rpc.interfaces.RiskScorerService'
    # version = '1.0.0'
    # group = 'risk'
    
    # Zookeeper registry
    registry = ZookeeperRegistry('192.168.1.194:2181')
    
    # Start Dubbo server
    server = DubboServer(
        service=service,
        interface=interface,
        # version=version,
        # group=group,
        registry=registry,
        codec=DubboCodec(),
        dubbo_version=DubboVersion.DEFAULT
    )
    server.start()
