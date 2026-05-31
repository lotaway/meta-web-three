package com.metawebthree.riskcontrol.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.riskcontrol.domain.RiskEvent;
import com.metawebthree.riskcontrol.domain.RiskRule;
import com.metawebthree.riskcontrol.repository.RiskEventRepository;
import com.metawebthree.riskcontrol.repository.RiskRuleRepository;
import org.apache.dubbo.config.annotation.DubboService;
import org.springframework.beans.factory.annotation.Autowired;

import java.util.List;
import java.util.UUID;

@DubboService
public class RiskControlService {

    @Autowired
    private RiskEventRepository riskEventRepository;

    @Autowired
    private RiskRuleRepository riskRuleRepository;

    public RiskEvent assessTransactionRisk(String userId, String scene, String orderId, Double amount) {
        List<RiskRule> rules = riskRuleRepository.selectList(
            new LambdaQueryWrapper<RiskRule>()
                .eq(RiskRule::getScene, scene)
                .eq(RiskRule::getStatus, 1)
                .orderByAsc(RiskRule::getPriority)
        );

        int totalScore = 0;
        StringBuilder details = new StringBuilder();

        for (RiskRule rule : rules) {
            if (evaluateRule(rule, amount)) {
                totalScore += rule.getScore();
                details.append(rule.getRuleName()).append("; ");
            }
        }

        RiskEvent event = new RiskEvent();
        event.setEventId(UUID.randomUUID().toString());
        event.setUserId(userId);
        event.setScene(scene);
        event.setEventType("TRANSACTION_RISK");
        event.setRiskScore(totalScore);
        event.setRiskLevel(calculateRiskLevel(totalScore));
        event.setDecision(calculateDecision(totalScore));
        event.setDetails(details.toString());
        event.setStatus(1);
        event.setCreateTime(System.currentTimeMillis());
        event.setUpdateTime(System.currentTimeMillis());
        event.setDeleted(0);

        riskEventRepository.insert(event);
        return event;
    }

    public RiskEvent detectAnomaly(String userId, String scene, String behaviorData) {
        int anomalyScore = calculateAnomalyScore(behaviorData);

        RiskEvent event = new RiskEvent();
        event.setEventId(UUID.randomUUID().toString());
        event.setUserId(userId);
        event.setScene(scene);
        event.setEventType("ANOMALY_DETECTION");
        event.setRiskScore(anomalyScore);
        event.setRiskLevel(calculateRiskLevel(anomalyScore));
        event.setDecision(calculateDecision(anomalyScore));
        event.setDescription("Anomaly detection triggered");
        event.setDetails(behaviorData);
        event.setStatus(1);
        event.setCreateTime(System.currentTimeMillis());
        event.setUpdateTime(System.currentTimeMillis());
        event.setDeleted(0);

        riskEventRepository.insert(event);
        return event;
    }

    public RiskEvent detectFraud(String userId, String scene, String orderId, List<String> fraudIndicators) {
        int fraudScore = calculateFraudScore(fraudIndicators);

        RiskEvent event = new RiskEvent();
        event.setEventId(UUID.randomUUID().toString());
        event.setUserId(userId);
        event.setScene(scene);
        event.setEventType("FRAUD_DETECTION");
        event.setRiskScore(fraudScore);
        event.setRiskLevel(calculateRiskLevel(fraudScore));
        event.setDecision(calculateDecision(fraudScore));
        event.setDescription("Fraud detection with " + fraudIndicators.size() + " indicators");
        event.setDetails(String.join(", ", fraudIndicators));
        event.setStatus(1);
        event.setCreateTime(System.currentTimeMillis());
        event.setUpdateTime(System.currentTimeMillis());
        event.setDeleted(0);

        riskEventRepository.insert(event);
        return event;
    }

    private boolean evaluateRule(RiskRule rule, Double amount) {
        String condition = rule.getCondition();
        if (condition == null) {
            return false;
        }
        
        if (condition.startsWith("amount>")) {
            double threshold = Double.parseDouble(condition.substring(7));
            return amount > threshold;
        }
        
        if (condition.startsWith("amount<")) {
            double threshold = Double.parseDouble(condition.substring(7));
            return amount < threshold;
        }
        
        return false;
    }

    private int calculateAnomalyScore(String behaviorData) {
        int score = 0;
        if (behaviorData != null) {
            if (behaviorData.contains("unusual_time")) score += 30;
            if (behaviorData.contains("rapid_sequence")) score += 40;
            if (behaviorData.contains("multiple_failures")) score += 50;
        }
        return Math.min(score, 100);
    }

    private int calculateFraudScore(List<String> indicators) {
        int baseScore = indicators.size() * 20;
        return Math.min(baseScore, 100);
    }

    private String calculateRiskLevel(int score) {
        if (score >= 70) return "HIGH";
        if (score >= 40) return "MEDIUM";
        return "LOW";
    }

    private String calculateDecision(int score) {
        if (score >= 70) return "REJECT";
        if (score >= 40) return "REVIEW";
        return "PASS";
    }
}