package com.metawebthree.service.repository;

import java.util.Map;

public interface ModelScorer {
    int score(String scene, Map<String, Object> features);
}
