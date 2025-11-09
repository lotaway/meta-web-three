#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gRPC Risk Scoring Service Implementation
Replaces the original Dubbo implementation with standard gRPC
"""

from random import Random
import grpc
import os
import numpy as np
import onnxruntime as ort
from RiskScorerService_pb2_grpc import (
    RiskScorerServiceServicer
)
from RiskScorerService_pb2 import (
    TestRequest, TestResponse,
    ScoreRequest, ScoreResponse
)
from Logger import init_logger

logger = init_logger()

class RiskScorerServiceImpl(RiskScorerServiceServicer):
    """
    gRPC Risk Scoring Service Implementation
    Inherits from auto-generated RiskScorerServiceServicer
    """
    
    def __init__(self):
        self.sess = None
        self._init_model()
    
    def _init_model(self):
        """Initialize machine learning model"""
        try:
            # Try to load ONNX model
            model_path = "risk_model.onnx"
            if os.path.exists(model_path):
                self.sess = ort.InferenceSession(
                    model_path, 
                    providers=["CPUExecutionProvider"]
                )
                logger.info("ONNX model loaded successfully")
            else:
                logger.warning("ONNX model not found, using fallback scoring")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            self.sess = None
    
    def test(self, request: TestRequest, context) -> TestResponse:
        """Test method implementation"""
        try:
            logger.info("Test method called")
            return TestResponse(result=Random().randint(1000, 5000))
        except Exception as e:
            logger.error(f"Test method error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {str(e)}")
            return TestResponse()
    
    def score(self, request: ScoreRequest, context) -> ScoreResponse:
        """Risk scoring method implementation"""
        try:
            logger.info(f"Score method called for scene: {request.scene}")
            
            # Extract feature data from request
            features = {}
            for key, feature in request.features.items():
                if feature.HasField('age'):
                    features['age'] = feature.age
                elif feature.HasField('external_debt_ratio'):
                    features['external_debt_ratio'] = feature.external_debt_ratio
                elif feature.HasField('first_order'):
                    features['first_order'] = feature.first_order
                elif feature.HasField('gps_stability'):
                    features['gps_stability'] = feature.gps_stability
                elif feature.HasField('device_shared_degree'):
                    features['device_shared_degree'] = feature.device_shared_degree
            
            # Calculate risk score
            score = self._calculate_risk_score(request.scene, features)
            
            logger.info(f"Risk score calculated: {score}")
            return ScoreResponse(score=score)
            
        except Exception as e:
            logger.error(f"Score method error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {str(e)}")
            return ScoreResponse()
    
    def _calculate_risk_score(self, scene: str, features: dict) -> float:
        """
        Calculate risk score
        Prioritize machine learning model, fallback to rule engine
        """
        try:
            # Try to use ONNX model
            if self.sess is not None:
                return self._predict_with_model(features)
        except Exception as e:
            logger.warning(f"Model prediction failed: {e}")
        
        # Use rule engine as fallback
        return self._calculate_with_rules(features)
    
    def _predict_with_model(self, features: dict) -> float:
        """Use machine learning model for prediction"""
        try:
            # Build feature vector
            feature_vector = self._build_feature_vector(features)
            x = np.array([feature_vector], dtype=np.float32)
            
            # Run inference
            result = self.sess.run(None, {"input": x})[0]
            prob = np.array(result)[0][0][1].item()
            
            # Convert probability to score (800-500 range)
            score = 800 - prob * 300
            return max(300, min(900, score))
            
        except Exception as e:
            logger.error(f"Model prediction error: {e}")
            raise
    
    def _build_feature_vector(self, features: dict) -> list:
        """Build feature vector"""
        # Default values
        default_values = {
            'age': 30,
            'external_debt_ratio': 0.0,
            'first_order': False,
            'gps_stability': 0.5,
            'device_shared_degree': 1
        }
        
        # Build feature vector
        vector = []
        for key in ['age', 'external_debt_ratio', 'first_order', 'gps_stability', 'device_shared_degree']:
            value = features.get(key, default_values[key])
            if key == 'first_order':
                vector.append(1.0 if value else 0.0)
            else:
                vector.append(float(value))
        
        return vector
    
    def _calculate_with_rules(self, features: dict) -> float:
        """Calculate score using rule engine"""
        base_score = 800.0
        
        # Age rules
        age = features.get('age', 30)
        if age < 25:
            base_score -= 50
        elif age > 60:
            base_score -= 30
        
        # Debt ratio rules
        debt_ratio = features.get('external_debt_ratio', 0.0)
        base_score -= debt_ratio * 200
        
        # First order rules
        if features.get('first_order', False):
            base_score += 20
        
        # GPS stability rules
        gps_stability = features.get('gps_stability', 0.5)
        base_score += gps_stability * 50
        
        # Device shared degree rules
        device_shared_degree = features.get('device_shared_degree', 1)
        base_score -= device_shared_degree * 30
        
        # Ensure score is within reasonable range
        return max(300, min(900, base_score))
