package com.metawebthree.service.repository;

import com.metawebthree.dto.DecisionRequest;
import java.util.List;
import java.util.Map;

public interface AuditRepo {
    void save(DecisionRequest request, Map<String, Object> features, 
              String decision, int score, List<String> reasons);
}
