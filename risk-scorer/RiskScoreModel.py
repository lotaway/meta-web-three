import dubbo
from dubbo.configs import ApplicationConfig, RegistryConfig, ServiceConfig, ReferenceConfig
from dubbo.proxy.handlers import RpcMethodHandler, RpcServiceHandler
from dubbo.bootstrap import Dubbo
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

class RiskScorerServiceImpl:
    
    def init(cls):
        # @TODO get risk model
        if (cls.sess is None):
            cls.sess = ort.InferenceSession('risk_model.onnx', providers=['CPUExecutionProvider'])

    def test(self,  _=None):
        return 100

    def score(self, scene, features):
        """Calculate risk score based on input features"""
        RiskScorerServiceImpl.init()
        try:
            # Prepare input data
            x = np.array([features['values']], dtype=np.float32)
            
            # Run inference
            result = RiskScorerServiceImpl.sess.run(None, {'input': x})[0]
            prob = np.array(result)[0][0][1].item()
            
            # Convert probability to score (800-500 range)
            score = int(800 - prob * 300)
            return score
        except Exception as e:
            # Fallback score calculation
            debt = features.get('external_debt_ratio', 0)
            age = features.get('age', 30)
            return int(700 - debt * 120 - max(0, 25 - age) * 2)
        
def build_service_handler():
    # build a method handler
    test_method_handler = RpcMethodHandler.unary(
        method=RiskScorerServiceImpl().test, method_name="test"
    )
    score_method_handler = RpcMethodHandler.unary(
        method=RiskScorerServiceImpl().score, method_name="score"
    )
    # build a service handler
    interface = os.getenv("SERVICE_PACKAGE_NAME") or "com.metawebthree.common.rpc.interfaces.RiskScorerService"
    service_handler = RpcServiceHandler(
        service_name=interface,
        method_handlers=[test_method_handler, score_method_handler],
    )
    return service_handler

def start_risk_score_model():
    # Configure the Zookeeper registry
    zk = os.getenv("ZK_HOST")
    application_config = ApplicationConfig("risk-scorer-service")
    registry_config = RegistryConfig.from_url(f"zookeeper://{zk}/dubbo")
    # bootstrap = Dubbo(registry_config=registry_config)

    # Create the client
    # client = bootstrap.create_client(reference_config)
    
    # build service config
    service_handler = build_service_handler()
    service_config = ServiceConfig(
        service_handler=service_handler, port=(os.getenv("EXPORT_DUBBO_PORT") or 20088),
        # protocol="dubbo", # not support dubbo?
    )
    dubbo_config = Dubbo(
        application_config,
        registry_config,
    )
    # Create and start the server
    # bootstrap.create_server(service_config).start()
    # start the server
    server = dubbo.Server(
        service_config, 
        dubbo_config,
    )
    server.start()
    return server

