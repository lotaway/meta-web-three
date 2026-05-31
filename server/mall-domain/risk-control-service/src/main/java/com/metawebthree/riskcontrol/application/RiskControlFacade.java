package com.metawebthree.riskcontrol.application;

import com.metawebthree.riskcontrol.domain.RiskEvent;
import com.metawebthree.riskcontrol.service.RiskControlService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class RiskControlFacade {

    @Autowired
    private RiskControlService riskControlService;

    public RiskEvent assessOrderRisk(String userId, String orderId, Double amount) {
        return riskControlService.assessTransactionRisk(userId, "ORDER", orderId, amount);
    }

    public RiskEvent assessPaymentRisk(String userId, String paymentId, Double amount) {
        return riskControlService.assessTransactionRisk(userId, "PAYMENT", paymentId, amount);
    }

    public RiskEvent detectUserAnomaly(String userId, String behaviorData) {
        return riskControlService.detectAnomaly(userId, "USER_BEHAVIOR", behaviorData);
    }

    public RiskEvent checkFraud(String userId, String orderId, List<String> indicators) {
        return riskControlService.detectFraud(userId, "FRAUD_CHECK", orderId, indicators);
    }
}