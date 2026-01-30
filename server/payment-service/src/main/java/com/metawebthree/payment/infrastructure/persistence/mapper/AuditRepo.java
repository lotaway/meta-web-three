package com.metawebthree.payment.infrastructure.persistence.mapper;
import com.metawebthree.payment.domain.model.*;

import com.metawebthree.dto.DecisionRequest;
import com.metawebthree.enums.DecisionEnum;

import java.util.List;
import java.util.Map;

public interface AuditRepo {
    void save(DecisionRequest request, Map<String, Object> features, 
              DecisionEnum decision, int score, List<String> reasons);
}
