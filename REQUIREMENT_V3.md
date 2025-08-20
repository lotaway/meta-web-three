一、核心功能需求（按业务链路）

1）身份与设备
KYC 实名与四要素核验
设备指纹与多设备共用识别
黑产库、法院执行、欺诈名单命中

2）新客授信与提额
多源征信与行为特征聚合
授信评分卡与拒绝规则
额度策略与动态提额

3）交易与用信
实时交易反欺诈
交易限额与风控拦截
支付渠道风控联动与 3DS/二次校验

4）贷后与逾期
早期预警与分层催收
代扣失败智能重试与节假日策略
滚动违约率与Vintage看板

5）运营与合规
规则灰度与A/B
特征血缘与审计追溯
模型可解释与拒贷原因枚举
数据最小化与脱敏留痕

二、实时决策流（标准版）

请求 → 特征装载 → 规则引擎 → 模型打分 → 策略编排 → 审计落库与事件投递 → 响应
超时控制：全链路 P99 ≤ 50ms（本地特征）/ ≤ 120ms（含外部征信）

三、关键数据与埋点

主键：user_id、device_id、biz_order_id、idfa/android_id、bank_card_hash
高频特征：注册时长、设备共享度、历史逾期标签、订单转化率、支付失败率、GPS稳定度、同群体欺诈率
事件：register, apply, approve, reject, lend, repay, overdue, chargeoff

四、策略示例（先规则、后模型、再编排）

拒绝规则优先生效 → 模型阈值 → 额度定价 → 二次校验 → 审批结果与原因码

示例拒绝原因码：R101 身份校验失败、R203 设备高危、R305 多平台负债过高、M402 模型阈值拒绝

五、数据表设计（MySQL）
CREATE TABLE risk_decision_log (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  biz_order_id VARCHAR(64),
  user_id VARCHAR(64),
  device_id VARCHAR(128),
  scene VARCHAR(32),
  decision VARCHAR(16),
  score INT,
  reasons JSON,
  features JSON,
  latency_ms INT,
  ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE risk_rules (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  scene VARCHAR(32),
  rule_code VARCHAR(32),
  expr TEXT,
  priority INT,
  status TINYINT,
  version VARCHAR(16),
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE credit_profile (
  user_id VARCHAR(64) PRIMARY KEY,
  credit_limit INT,
  credit_used INT,
  risk_level VARCHAR(16),
  last_score INT,
  last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

六、事件与主题（Kafka）

topic：risk.events
apply、approve、reject、intercept、overdue、repay

消息体

{
  "event":"apply",
  "biz_order_id":"B202508200001",
  "user_id":"U123",
  "device_id":"Dabc",
  "scene":"new_credit",
  "decision":"review",
  "score":642,
  "reasons":["M402"],
  "ts":1724123456789
}

七、规则表达示例（JSON DSL）
[
  {
    "code":"R203",
    "when":"device_shared_degree > 5 and device_risk_tag in ['emu','root','hook']",
    "action":"reject",
    "priority":10
  },
  {
    "code":"R305",
    "when":"external_debt_ratio >= 0.8",
    "action":"reject",
    "priority":20
  },
  {
    "code":"W101",
    "when":"first_order and gps_stability < 0.3",
    "action":"review",
    "priority":100
  }
]

八、实时决策服务（Java Spring Boot，规则+模型+编排）
package com.hello.risk

import org.springframework.boot.SpringApplication
import org.springframework.boot.autoconfigure.SpringBootApplication

@SpringBootApplication
public class RiskApplication {
  public static void main(String[] args) {
    SpringApplication.run(RiskApplication.class, args)
  }
}

package com.hello.risk.api

import org.springframework.web.bind.annotation.*
import org.springframework.beans.factory.annotation.Autowired

@RestController
@RequestMapping("/risk")
public class DecisionController {

  @Autowired
  private DecisionService decisionService

  @PostMapping("/decision")
  public DecisionResponse decide(@RequestBody DecisionRequest req) {
    return decisionService.decide(req)
  }
}

package com.hello.risk.core

import org.springframework.stereotype.Service
import java.util.*
import javax.script.ScriptEngineManager

@Service
public class DecisionService {

  private final RuleRepo ruleRepo = new RuleRepo()
  private final FeatureRepo featureRepo = new FeatureRepo()
  private final ModelScorer modelScorer = new ModelScorer()
  private final AuditRepo auditRepo = new AuditRepo()

  public DecisionResponse decide(DecisionRequest req) {
    Map<String,Object> feats = featureRepo.load(req)
    List<Rule> rules = ruleRepo.load(req.scene)
    List<String> reasons = new ArrayList<>()
    for (Rule r : rules) {
      if (eval(r.expr, feats)) {
        if ("reject".equals(r.action)) return audit(req, feats, "reject", 0, Arrays.asList(r.code))
        if ("review".equals(r.action)) reasons.add(r.code)
      }
    }
    int score = modelScorer.score(req.scene, feats)
    if (score < 620) return audit(req, feats, "reject", score, Arrays.asList("M402"))
    String decision = reasons.isEmpty() ? "approve" : "review"
    return audit(req, feats, decision, score, reasons)
  }

  private boolean eval(String expr, Map<String,Object> feats) {
    ScriptEngineManager m = new ScriptEngineManager()
    javax.script.ScriptEngine e = m.getEngineByName("nashorn")
    for (Map.Entry<String,Object> kv : feats.entrySet()) e.put(kv.getKey(), kv.getValue())
    Object v = e.eval(expr)
    return v instanceof Boolean ? (Boolean)v : false
  }

  private DecisionResponse audit(DecisionRequest req, Map<String,Object> feats, String decision, int score, List<String> reasons) {
    auditRepo.save(req, feats, decision, score, reasons)
    return new DecisionResponse(decision, score, reasons)
  }
}

package com.hello.risk.core

public class DecisionRequest {
  public String bizOrderId
  public String userId
  public String deviceId
  public String scene
  public Map<String,Object> context
}

package com.hello.risk.core

public class DecisionResponse {
  public String decision
  public int score
  public java.util.List<String> reasons
  public DecisionResponse(String d, int s, java.util.List<String> r) {
    this.decision = d
    this.score = s
    this.reasons = r
  }
}

package com.hello.risk.core

import java.util.*

public class FeatureRepo {
  public Map<String,Object> load(DecisionRequest req) {
    Map<String,Object> m = new HashMap<>()
    m.put("device_shared_degree", 6)
    m.put("device_risk_tag", "emu")
    m.put("external_debt_ratio", 0.45)
    m.put("first_order", true)
    m.put("gps_stability", 0.62)
    m.put("age", 28)
    return m
  }
}

package com.hello.risk.core

import java.util.*

public class Rule {
  public String code
  public String expr
  public String action
  public int priority
}

package com.hello.risk.core

import java.util.*

public class RuleRepo {
  public List<Rule> load(String scene) {
    List<Rule> list = new ArrayList<>()
    Rule r1 = new Rule()
    r1.code = "R203"
    r1.expr = "device_shared_degree > 5 && device_risk_tag == 'emu'"
    r1.action = "reject"
    r1.priority = 10
    list.add(r1)
    Rule r2 = new Rule()
    r2.code = "W101"
    r2.expr = "first_order && gps_stability < 0.3"
    r2.action = "review"
    r2.priority = 100
    list.add(r2)
    return list
  }
}

package com.hello.risk.core

public class ModelScorer {
  public int score(String scene, java.util.Map<String,Object> feats) {
    Object debt = feats.get("external_debt_ratio")
    double d = debt instanceof Number ? ((Number)debt).doubleValue() : 0d
    Object age = feats.get("age")
    int a = age instanceof Number ? ((Number)age).intValue() : 30
    double s = 700 - d * 120 - Math.max(0, 25 - a) * 2
    return (int)Math.round(s)
  }
}

package com.hello.risk.core

public class AuditRepo {
  public void save(DecisionRequest req, java.util.Map<String,Object> feats, String decision, int score, java.util.List<String> reasons) {
  }
}

九、模型服务（Python，ONNX 推理与HTTP打分）

训练与导出示例

import numpy as np
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
X = np.random.rand(2000, 6)
y = (X[:,0]*1.2 + X[:,1]*0.8 + X[:,2]*1.5 > 1.8).astype(int)
clf = LogisticRegression(max_iter=200)
clf.fit(X, y)
onnx_model = convert_sklearn(clf, initial_types=[('input', FloatTensorType([None, 6]))])
with open('/mnt/data/risk_model.onnx','wb') as f:
    f.write(onnx_model.SerializeToString())


推理服务

from fastapi import FastAPI
import uvicorn
import onnxruntime as ort
import numpy as np

app = FastAPI()
sess = ort.InferenceSession('/mnt/data/risk_model.onnx', providers=['CPUExecutionProvider'])

@app.post('/score')
async def score(payload: dict):
    feats = payload.get('features', [])
    x = np.array([feats], dtype=np.float32)
    prob = sess.run(None, {'input': x})[0][0][1].item()
    score = int(800 - prob * 300)
    return {'score': score}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8081)

十、提额与定价策略

提额
近3月逾期为0、使用率>60%、交易成功率>95%、评分提升>20分，提额5%~15%，季度上限一次

定价
风险分层A/B/C/D，对应日息或费率区间，交易高危标记命中提升费率或二次校验

十一、贷后与逾期分层

M0 提前提醒
M1 智能外呼+短信组合
M2 人工介入与分案
M3 司法前提醒
滚动指标：DPD1+, DPD7+, M1回收率，滚动30日策略命中率

十二、灰度与验收

灰度开关按 user_id hash 分流
线上验收口径：拦截准确率、拒绝命中率、绝对坏账率变化、客损影响、响应时延

十三、最小可运行联调

1）启动 Java 风控服务
2）启动 Python 模型服务
3）发起请求

curl -X POST http://localhost:8080/risk/decision \
  -H 'Content-Type: application/json' \
  -d '{
    "bizOrderId":"B202508200001",
    "userId":"U123",
    "deviceId":"Dabc",
    "scene":"new_credit",
    "context":{"features":[0.2,0.3,0.1,0.4,0.5,0.6]}
  }'

十四、落地建议

优先落地新客授信与交易反欺诈两个场景
特征先本地可得，外部征信二期接入
规则先 JSON DSL，后续可切 Drools 或自研表达式
模型先单一评分卡或轻量二分类，再引入多模型编排