package com.metawebthree.service.repository.impl;

import com.metawebthree.dto.DecisionRequest;
import com.metawebthree.service.repository.AuditRepo;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Map;

@Slf4j
@Repository
public class AuditRepoImpl implements AuditRepo {

    @Override
    public void save(DecisionRequest request, Map<String, Object> features, 
                    String decision, int score, List<String> reasons) {
        log.info("Risk decision audit - bizOrderId: {}, userId: {}, deviceId: {}, scene: {}, " +
                "decision: {}, score: {}, reasons: {}, features: {}",
                request.getBizOrderId(), request.getUserId(), request.getDeviceId(), 
                request.getScene(), decision, score, reasons, features);
    }
}
