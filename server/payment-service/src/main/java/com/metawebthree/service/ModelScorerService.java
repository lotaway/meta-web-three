package com.metawebthree.service;

import java.util.Map;

public interface ModelScorerService {
    int score(String scene, Map<String, Object> features);
}
