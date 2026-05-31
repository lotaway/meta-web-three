package com.metawebthree.riskcontrol.interfaces;

import com.metawebthree.riskcontrol.application.RiskControlFacade;
import com.metawebthree.riskcontrol.domain.RiskEvent;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/risk")
public class RiskControlController {

    @Autowired
    private RiskControlFacade riskControlFacade;

    @PostMapping("/order/check")
    public Map<String, Object> checkOrderRisk(@RequestParam String userId,
                                               @RequestParam String orderId,
                                               @RequestParam Double amount) {
        RiskEvent event = riskControlFacade.assessOrderRisk(userId, orderId, amount);
        return buildResponse(event);
    }

    @PostMapping("/payment/check")
    public Map<String, Object> checkPaymentRisk(@RequestParam String userId,
                                                  @RequestParam String paymentId,
                                                  @RequestParam Double amount) {
        RiskEvent event = riskControlFacade.assessPaymentRisk(userId, paymentId, amount);
        return buildResponse(event);
    }

    @PostMapping("/anomaly/detect")
    public Map<String, Object> detectAnomaly(@RequestParam String userId,
                                              @RequestParam String behaviorData) {
        RiskEvent event = riskControlFacade.detectUserAnomaly(userId, behaviorData);
        return buildResponse(event);
    }

    @PostMapping("/fraud/check")
    public Map<String, Object> checkFraud(@RequestParam String userId,
                                           @RequestParam String orderId,
                                           @RequestParam(required = false) List<String> indicators) {
        RiskEvent event = riskControlFacade.checkFraud(userId, orderId, indicators);
        return buildResponse(event);
    }

    private Map<String, Object> buildResponse(RiskEvent event) {
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("eventId", event.getEventId());
        response.put("riskScore", event.getRiskScore());
        response.put("riskLevel", event.getRiskLevel());
        response.put("decision", event.getDecision());
        response.put("details", event.getDetails());
        return response;
    }
}