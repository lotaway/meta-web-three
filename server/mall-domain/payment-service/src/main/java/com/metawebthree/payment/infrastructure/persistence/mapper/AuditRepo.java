package com.metawebthree.payment.infrastructure.persistence.mapper;
import com.metawebthree.payment.domain.model.*;

import com.metawebthree.payment.application.dto.DecisionRequest;
import com.metawebthree.payment.enums.DecisionEnum;

import java.util.List;
import java.util.Map;

public interface AuditRepo {
    void save(DecisionRequest request, Map<String, Object> features, 
              DecisionEnum decision, int score, List<String> reasons);
}
