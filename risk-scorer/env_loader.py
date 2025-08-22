from easy_dotenv import EnvConfig

class BaseEnv(EnvConfig):
    # Base application settings
    zk_port: str = "localhost:2181"
    service_package_name: str = "com.metawebthree.common.rpc.interfaces.RiskScorerService"
    
base = BaseEnv('..')

__all__ = ['base']