package com.metawebthree.common.rpc.interfaces;

import java.util.Map;

public interface RiskScorerService {
    int test();
    int score(String scene, Map<String, Object> features);
}
