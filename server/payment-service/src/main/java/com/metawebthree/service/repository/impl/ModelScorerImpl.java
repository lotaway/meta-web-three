package com.metawebthree.service.repository.impl;

import com.metawebthree.service.repository.ModelScorer;
import org.springframework.stereotype.Repository;

import java.util.Map;

@Repository
public class ModelScorerImpl implements ModelScorer {

    @Override
    public int score(String scene, Map<String, Object> features) {
        Object debt = features.get("external_debt_ratio");
        double debtRatio = debt instanceof Number ? ((Number)debt).doubleValue() : 0d;
        
        Object age = features.get("age");
        int ageValue = age instanceof Number ? ((Number)age).intValue() : 30;
        
        double score = 700 - debtRatio * 120 - Math.max(0, 25 - ageValue) * 2;
        return (int) Math.round(score);
    }
}
