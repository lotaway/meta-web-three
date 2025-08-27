#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import grpc
import time
from concurrent import futures
import threading
from generated.rpc.RiskScorerService_pb2_grpc import (
    add_RiskScorerServiceServicer_to_server
)
from generated.rpc.RiskScorerService_pb2 import (
    TestRequest, ScoreRequest, Feature
)
from RiskScoreModel import RiskScorerServiceImpl


class TestRiskScorerGrpcService(unittest.TestCase):
    """gRPC 风险评分服务测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 启动测试服务器
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        add_RiskScorerServiceServicer_to_server(RiskScorerServiceImpl(), self.server)
        self.server.add_insecure_port('[::]:0')  # 使用随机端口
        self.server.start()
        
        # 获取实际端口
        self.port = self.server.add_insecure_port('[::]:0')
        self.server.remove_port(self.port)
        self.server.add_insecure_port(f'[::]:{self.port}')
        
        # 创建客户端通道
        self.channel = grpc.insecure_channel(f'localhost:{self.port}')
        
        # 等待服务器启动
        time.sleep(1)
    
    def tearDown(self):
        """测试后清理"""
        self.channel.close()
        self.server.stop(0)
    
    def test_test_method(self):
        """测试测试方法"""
        from generated.rpc.RiskScorerService_pb2_grpc import RiskScorerServiceStub
        
        stub = RiskScorerServiceStub(self.channel)
        request = TestRequest()
        response = stub.test(request)
        
        self.assertEqual(response.result, 100)
    
    def test_score_method(self):
        """测试评分方法"""
        from generated.rpc.RiskScorerService_pb2_grpc import RiskScorerServiceStub
        
        stub = RiskScorerServiceStub(self.channel)
        
        # 创建测试特征
        features = {
            'age': Feature(age=25),
            'external_debt_ratio': Feature(external_debt_ratio=0.1),
            'first_order': Feature(first_order=True),
            'gps_stability': Feature(gps_stability=0.8),
            'device_shared_degree': Feature(device_shared_degree=1)
        }
        
        request = ScoreRequest(
            scene='test_scene',
            features=features
        )
        
        response = stub.score(request)
        
        # 验证响应
        self.assertIsInstance(response.score, float)
        self.assertGreaterEqual(response.score, 300)
        self.assertLessEqual(response.score, 900)
    
    def test_score_method_with_missing_features(self):
        """测试缺少特征时的评分方法"""
        from generated.rpc.RiskScorerService_pb2_grpc import RiskScorerServiceStub
        
        stub = RiskScorerServiceStub(self.channel)
        
        # 只提供部分特征
        features = {
            'age': Feature(age=30)
        }
        
        request = ScoreRequest(
            scene='test_scene',
            features=features
        )
        
        response = stub.score(request)
        
        # 验证响应
        self.assertIsInstance(response.score, float)
        self.assertGreaterEqual(response.score, 300)
        self.assertLessEqual(response.score, 900)


def run_server_in_background():
    """在后台运行服务器"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    add_RiskScorerServiceServicer_to_server(RiskScorerServiceImpl(), server)
    server.add_insecure_port('[::]:9090')
    server.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    # 运行测试
    unittest.main()
