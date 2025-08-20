package com.metawebthree.repository;

import com.metawebthree.dto.DecisionRequest;
import java.util.Map;

public interface FeatureRepo {
    Map<String, Object> load(DecisionRequest request);
}
