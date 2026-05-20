package com.metawebthree.payment.infrastructure.persistence.mapper;

import com.metawebthree.payment.application.dto.DecisionRequest;
import java.util.Map;

public interface FeatureRepo {
    Map<String, Object> load(DecisionRequest request);
}
