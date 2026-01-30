package com.metawebthree.payment.infrastructure.persistence.mapper;
import com.metawebthree.payment.domain.model.*;

import com.metawebthree.dto.DecisionRequest;
import java.util.Map;

public interface FeatureRepo {
    Map<String, Object> load(DecisionRequest request);
}
