from concurrent import futures
import time
from tokenize import group
from kazoo.client import KazooClient
from kazoo.exceptions import NodeExistsError
import grpc
from RiskScorerService_pb2_grpc import (
    add_RiskScorerServiceServicer_to_server
)
from RiskScoreModel import RiskScorerServiceImpl
from Logger import init_logger
import os

logger = init_logger()

class ZookeeperServiceRegistry:
    """Zookeeper service registry for gRPC service discovery"""
    
    def __init__(self, zk_hosts, service_name, service_host, service_port, group_name):
        self.zk_hosts = zk_hosts
        self.group_name = self._final_service_group(group_name)
        self.service_name = service_name
        self.service_host = service_host
        self.service_port = service_port
        self.zk = None
        self.service_path = f"{group_name}/{service_name}/providers"
    
    def start(self):
        """Start Zookeeper connection and register service"""
        try:
            # Connect to Zookeeper
            self.zk = KazooClient(hosts=self.zk_hosts)
            self.zk.start()
            logger.info(f"Connected to Zookeeper at {self.zk_hosts}")
            
            # Create service path if not exists
            self.zk.ensure_path(self.service_path)
            
            # Register service with Dubbo-compatible URL format
            # Format: provider://host:port/service_name?interface=service_name&application=appName&category=providers&dubbo=2.0.2&release=3.3.5&side=provider&methods=methods&timestamp=timestamp
            import urllib.parse
            import time
            import os
            timestamp = int(time.time() * 1000)
            pid = os.getpid()
            methods = "score,scoreAsync,test,testAsync"
            provider_url = f"provider://{self.service_host}:{self.service_port}/{self.service_name}?interface={self.service_name}&application=risk-scorer&category=providers&dubbo=2.0.2&release=3.3.5&side=provider&methods={methods}&pid={pid}&timestamp={timestamp}"
            encoded_url = urllib.parse.quote(provider_url, safe='')
            node_path = f"{self.service_path}/{encoded_url}"
            self.zk.create(node_path, b"", ephemeral=True, makepath=True)
            logger.info(f"Service registered with Dubbo format: {provider_url}")
            
        except Exception as e:
            logger.error(f"Failed to register service with Zookeeper: {e}")
            raise

    def _final_service_group(self, group_name):
        if group_name == "" or group_name.startswith("/"):
            return group_name
        else:
            return f"/{group_name}"
    
    def stop(self):
        """Stop Zookeeper connection"""
        if self.zk:
            self.zk.stop()
            self.zk.close()
            logger.info("Zookeeper connection closed")


def start_risk_score_model():
    """Start gRPC server with Zookeeper registration"""
    # Get configuration from environment variables
    zk_hosts = os.getenv("ZK_HOST", "localhost:2181")
    service_host = os.getenv("SERVICE_HOST", "0.0.0.0")
    service_port = int(os.getenv("RPC_PORT", "20088"))
    # use fully-qualified gRPC service name
    service_name = os.getenv("SERVICE_NAME", "com.metawebthree.common.generated.rpc.RiskScorerService")
    group_name = os.getenv("GROUP_NAME", "")
    
    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_RiskScorerServiceServicer_to_server(RiskScorerServiceImpl(), server)
    
    # Listen on port
    listen_addr = f'[::]:{service_port}'
    server.add_insecure_port(listen_addr)
    
    # Start server
    logger.info(f"Starting gRPC server on {listen_addr}")
    server.start()
    
    # Register service with Zookeeper
    registry = ZookeeperServiceRegistry(zk_hosts, service_name, service_host, service_port, group_name)
    try:
        registry.start()
        logger.info("Service registered with Zookeeper successfully")
        
        # Keep server running
        while True:
            time.sleep(24 * 60 * 60)  # 24 hours
            
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server...")
        registry.stop()
        server.stop(0)
        logger.info("gRPC server stopped")
    except Exception as e:
        logger.error(f"Server error: {e}")
        registry.stop()
        server.stop(0)
        raise
