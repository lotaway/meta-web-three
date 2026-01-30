package com.metawebthree.payment.repository.impl;

import com.metawebthree.dto.DecisionRequest;
import com.metawebthree.enums.DeviceRiskTag;
import com.metawebthree.repository.FeatureRepo;

import org.springframework.stereotype.Repository;

import java.util.HashMap;
import java.util.Map;

@Repository
public class FeatureRepoImpl implements FeatureRepo {

    // @TODO use real data
    @Override
    public Map<String, Object> load(DecisionRequest request) {
        return mockData();
    }

    protected Map<String, Object> mockData() {
        Map<String, Object> features = new HashMap<>();
        // @TODO Move feature key into enum
        features.put("device_shared_degree", 6);
        features.put("device_risk_tag", DeviceRiskTag.EMU);
        features.put("external_debt_ratio", 0.45);
        features.put("first_order", true);
        features.put("gps_stability", 0.62);
        features.put("age", 28);

        return features;
    }
}
