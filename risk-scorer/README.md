## 风险评分微服务（risk-scorer）

- 提供可解释信用风险评分能力，支持评分卡（WOE+Logistic）与 ONNX 推理两种模式
- 遵循 DDD：业务仅依赖抽象接口，能力通过基础设施注入

## 快速开始

- 安装依赖
```bash
python3 -m pip install -r requiredment.txt
```

- 配置环境（复制 .env.example 为 .env，按需填写）
  - [.env.example](file:///Volumes/Extra/Projects/meta-web-three/risk-scorer/.env.example)
  - 关键变量：
    - RISK_MODEL_TYPE：joblib 或 onnx
    - RISK_MODEL_PATH：评分卡模型路径，默认 app/model/scorecard.pkl
    - RISK_ONNX_PATH：ONNX 模型路径
    - RISK_TARGET_LABEL：目标标签名，默认 default_flag
    - RISK_BASE_SCORE、RISK_PDO、RISK_APPROVE_THRESHOLD、RISK_REVIEW_THRESHOLD：分数映射与决策阈值
    - ZK_HOST、SERVICE_HOST、RPC_PORT、SERVICE_NAME、GROUP_NAME、PORT：服务注册与端口

- 启动服务
```bash
python3 main.py
```

- 入口脚本：[main.py](file:///Volumes/Extra/Projects/meta-web-three/risk-scorer/main.py)
- gRPC 服务实现：[risk_scorer_grpc_service.py](file:///Volumes/Extra/Projects/meta-web-three/risk-scorer/app/risk_scorer_grpc_service.py)

## 评分卡生成

- 从 DataFrame 训练并保存评分卡
  - 训练脚本：[scorecard_trainer.py](file:///Volumes/Extra/Projects/meta-web-three/risk-scorer/app/training/scorecard_trainer.py)

```python
import os
import pandas as pd
from app.training.scorecard_trainer import train_from_dataframe, save_payload

os.environ["RISK_MODEL_PATH"] = "app/model/scorecard.pkl"
df = pd.read_csv("your_dataset.csv")  # 包含目标列，如 default_flag
payload = train_from_dataframe(df, "default_flag")
save_payload(payload)
```

- 设置使用评分卡模式
```bash
export RISK_MODEL_TYPE=joblib
export RISK_MODEL_PATH=app/model/scorecard.pkl
```

## 使用 ONNX 模型

- 设置使用 ONNX 推理
```bash
export RISK_MODEL_TYPE=onnx
export RISK_ONNX_PATH=risk_model.onnx
```

- ONNX 预测适配器：[predictor_onnx.py](file:///Volumes/Extra/Projects/meta-web-three/risk-scorer/app/infrastructure/predictor_onnx.py)

## gRPC 接口

- Proto 文件：[RiskScorerService.proto](file:///Volumes/Extra/Projects/meta-web-three/protos/RiskScorerService.proto)
- 生成 Python Stub 示例
```bash
python3 -m grpc_tools.protoc \
  -I ../../protos \
  --python_out=. \
  --grpc_python_out=. \
  ../../protos/RiskScorerService.proto
```

- Python 请求示例
```python
import grpc
from RiskScorerService_pb2 import ScoreRequest, Feature
from RiskScorerService_pb2_grpc import RiskScorerServiceStub

channel = grpc.insecure_channel("localhost:20088")
stub = RiskScorerServiceStub(channel)

features = {
  "age": Feature(age=30),
  "external_debt_ratio": Feature(external_debt_ratio=0.1),
  "first_order": Feature(first_order=True),
  "gps_stability": Feature(gps_stability=0.7),
  "device_shared_degree": Feature(device_shared_degree=1),
}

resp = stub.score(ScoreRequest(scene="apply", features=features))
print(resp.score, resp.decision)
```

## 测试

- 运行测试
```bash
python3 -m pytest -q
```

- 测试位置：
  - 领域策略测试：[test_policy.py](file:///Volumes/Extra/Projects/meta-web-three/risk-scorer/tests/test_policy.py)
  - 用例行为测试：[test_usecase.py](file:///Volumes/Extra/Projects/meta-web-three/risk-scorer/tests/test_usecase.py)
  - 训练脚本测试：[test_training.py](file:///Volumes/Extra/Projects/meta-web-three/risk-scorer/tests/test_training.py)

## 目录与职责

- 入口与注册：[main.py](file:///Volumes/Extra/Projects/meta-web-three/risk-scorer/main.py)、[grpcClient.py](file:///Volumes/Extra/Projects/meta-web-three/risk-scorer/grpcClient.py)
- 应用层与领域层：
  - 用例编排：[scoring_usecase.py](file:///Volumes/Extra/Projects/meta-web-three/risk-scorer/app/application/scoring_usecase.py)
  - 分数与决策策略：[scoring_policy.py](file:///Volumes/Extra/Projects/meta-web-three/risk-scorer/app/domain/scoring_policy.py)
  - 预处理：[preprocess.py](file:///Volumes/Extra/Projects/meta-web-three/risk-scorer/app/application/preprocess.py)
- 基础设施：
  - 分箱/WOE：[binning_scorecardpy.py](file:///Volumes/Extra/Projects/meta-web-three/risk-scorer/app/infrastructure/binning_scorecardpy.py)
  - 模型存储：[model_store_joblib.py](file:///Volumes/Extra/Projects/meta-web-three/risk-scorer/app/infrastructure/model_store_joblib.py)
  - 模型工厂：[model_store_factory.py](file:///Volumes/Extra/Projects/meta-web-three/risk-scorer/app/infrastructure/model_store_factory.py)
  - ONNX 预测：[predictor_onnx.py](file:///Volumes/Extra/Projects/meta-web-three/risk-scorer/app/infrastructure/predictor_onnx.py)
