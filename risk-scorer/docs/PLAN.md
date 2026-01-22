1. 阶段 1 — 数据探索与清洗

目标

理解原始字段分布、缺失、类别分布

明确 target label 与样本定义

关键决策点

确定违约定义与观察窗口

是否做拒绝推断（Reject Inference）补偿样本偏差

策略说明

用样本视觉化工具探索异常值

对离群值/不可解释值做业务规则处理

2. 阶段 2 — 分箱策略

目标

对连续特征做分箱，便于生成单调风险类别

关键点

初始分箱（如决策树分箱/百分位分箱体系）

粗分箱后检查每箱坏客户比例单调性

策略

使用 Python 分箱库（如 optbinning 或自定义）

检验每箱的 WOE 值单调趋势（这是评分卡可解释的基础）

边界决策

若分箱无法满足单调趋势，则尝试调整分界或组合箱

3. 阶段 3 — WOE 与 IV

目标

转换分箱为 WOE，计算 IV，用于后续特征选择

实施说明

每变量按分箱组计算 WOE & IV

根据 IV 排序剔除低信息变量（如 IV < 0.02）

4. 阶段 4 — 模型训练

模型技术选择

Logistic 回归为主评分模型（风险可解释、符合行业要求）

训练要点

训练集/验证集切分

检查多重共线性（VIF）并剔除高相关变量

5. 阶段 5 — 评分卡构造

目标

将概率转换为标准分数

核心方法

使用对数概率转换公式设定基准点和 PDO（points to double odds）

调整评分区间范围

6. 阶段 6 — 模型评估与验收

评估指标

AUC / KS 评价分离能力

PSI 监控样本稳定性

验收策略

生成 PDF/报告说明所有指标、风险解释及最终策略

7. 线上策略与决策输出

目的

根据不同评分段输出不同业务策略

例

分数 [750+] → 直接审批

分数 [650-750) → 复审

分数 <650 → 拒绝

8. 解释与文档自动化

关键说明

输出变量贡献表 & SHAP/业务解释

输出“为什么拒绝”原因模板（可用 LLM 生成）

9. 关键风险与策略
风险	对策
分箱不稳定	手动调整或合并边缘箱
过拟合	增加正则化或简化特征
业务不可解释	调整模型特征与梳理说明


目录结构：

```makefile
risk-scorer-python/
├── protos/
│   └── risk_scorer.proto
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── model/
│   │   ├── scorer_model.py
│   │   └── scorecard.pkl        # 打包后的模型（训练后生成）
│   ├── grpc_server.py
│   ├── service_impl.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── binning.py
│   ├── train.py
│   └── evaluate.py
├── tests/
│   ├── test_scorer.py
├── data/
│   ├── raw/
│   └── processed/
├── requirements.txt
├── Dockerfile
├── Makefile
└── README.md

```

关键代码：

app/config.py
```python
MODEL_PATH = "app/model/scorecard.pkl"
TARGET = "default_flag"
```

🧠 app/scorer_model.py
```python
import joblib
from config import MODEL_PATH

model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
bins = model_data["bins"]

def predict_woe(df):
    from binning import apply_woe
    df_woe = apply_woe(df, bins)
    return model.predict_proba(df_woe.values)[0][1]
```

🧠 app/preprocess.py
```python
import pandas as pd

def to_dataframe(features):
    df = pd.DataFrame([features])
    df = df.apply(pd.to_numeric, errors="ignore")
    return df
```

🧠 app/binning.py
```python
import scorecardpy as sc

def generate_bins(df, target):
    return sc.woebin(df, y=target)

def apply_woe(df, bins):
    return sc.woebin_ply(df, bins)
```

🧠 app/train.py
```python
from sklearn.linear_model import LogisticRegression
import joblib
from preprocess import to_dataframe
from binning import generate_bins, apply_woe

def train(df, target):
    bins = generate_bins(df, target)
    df_woe = apply_woe(df, bins)
    X = df_woe.drop(columns=[target])
    y = df_woe[target]
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    joblib.dump({"model":model, "bins":bins}, "app/model/scorecard.pkl")
```

🧠 app/grpc_server.py
```python
from concurrent import futures
import grpc
import service_impl
import risk_scorer_pb2_grpc

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    risk_scorer_pb2_grpc.add_RiskScorerServicer_to_server(service_impl.RiskScorerServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
```

🧠 app/service_impl.py
```python
import risk_scorer_pb2
import risk_scorer_pb2_grpc
from scorer_model import predict_woe
from preprocess import to_dataframe

class RiskScorerServicer(risk_scorer_pb2_grpc.RiskScorerServicer):
    def Score(self, request, context):
        features = dict(request.features)
        df = to_dataframe(features)
        pd = predict_woe(df)
        score = 600 - 50 * (pd/(1-pd))
        decision = "approve" if score >= 700 else "review" if score >= 600 else "reject"
        return risk_scorer_pb2.ScoreResponse(score=score, decision=decision)
```

🧠 app/evaluate.py
```python
import scorecardpy as sc

def evaluate_model(model, df_woe, y):
    return sc.perf_eva(y, model.predict_proba(df_woe)[:,1])
```

📄 requirements.txt
```txt
grpcio
grpcio-tools
pandas
scikit-learn
scorecardpy
joblib
```

📄 Dockerfile
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "app/grpc_server.py"]
```

🧪 测试示例 tests/test_scorer.py
```python
from app.preprocess import to_dataframe
from app.service_impl import RiskScorerServicer
import risk_scorer_pb2

svc = RiskScorerServicer()

def test_score():
    req = risk_scorer_pb2.UserRequest(
        features={"age":"30","income":"10000"}
    )
    resp = svc.Score(req, None)
    assert isinstance(resp.score, float)
```
