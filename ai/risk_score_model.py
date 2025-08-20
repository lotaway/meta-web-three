from fastapi import FastAPI
import uvicorn
import onnxruntime as ort
import numpy as np
from pydantic import BaseModel
from typing import List

app = FastAPI()

# @TODO Load ONNX model for risk/credit scoring
sess = ort.InferenceSession('risk_model.onnx', providers=['CPUExecutionProvider'])

class RiskFeatures(BaseModel):
    features: List[float]
    scene: str

@app.post('/score')
async def score(payload: RiskFeatures):
    """Calculate risk score based on input features"""
    try:
        # Prepare input data
        x = np.array([payload.features], dtype=np.float32)
        
        # Run inference
        prob = sess.run(None, {'input': x})[0][0][1].item()
        
        # Convert probability to score (800-500 range)
        score = int(800 - prob * 300)
        
        return {
            'success': True,
            'score': score,
            'model_version': '1.0'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8081)
