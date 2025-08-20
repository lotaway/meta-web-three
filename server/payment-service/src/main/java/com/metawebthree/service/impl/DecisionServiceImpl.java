package com.metawebthree.service.impl;

import com.metawebthree.dto.DecisionRequest;
import com.metawebthree.dto.DecisionResponse;
import com.metawebthree.service.DecisionService;
import com.metawebthree.service.repository.AuditRepo;
import com.metawebthree.service.repository.FeatureRepo;
import com.metawebthree.service.repository.ModelScorer;
import com.metawebthree.service.repository.RuleRepo;
import com.metawebthree.service.entity.Rule;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import javax.script.ScriptEngineManager;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class DecisionServiceImpl implements DecisionService {

    private final RuleRepo ruleRepo;
    private final FeatureRepo featureRepo;
    private final ModelScorer modelScorer;
    private final AuditRepo auditRepo;

    @Override
    public DecisionResponse decide(DecisionRequest request) {
        Map<String, Object> features = featureRepo.load(request);
        List<Rule> rules = ruleRepo.load(request.getScene());
        List<String> reasons = new ArrayList<>();

        for (Rule rule : rules) {
            if (eval(rule.getExpr(), features)) {
                if ("reject".equals(rule.getAction())) {
                    return audit(request, features, "reject", 0, Arrays.asList(rule.getCode()));
                }
                if ("review".equals(rule.getAction())) {
                    reasons.add(rule.getCode());
                }
            }
        }

        int score = modelScorer.score(request.getScene(), features);
        if (score < 620) {
            return audit(request, features, "reject", score, Arrays.asList("M402"));
        }

        String decision = reasons.isEmpty() ? "approve" : "review";
        return audit(request, features, decision, score, reasons);
    }

    private boolean eval(String expr, Map<String, Object> features) {
        ScriptEngineManager manager = new ScriptEngineManager();
        var engine = manager.getEngineByName("nashorn");
        features.forEach(engine::put);
        try {
            Object result = engine.eval(expr);
            return result instanceof Boolean ? (Boolean) result : false;
        } catch (Exception e) {
            return false;
        }
    }

    private DecisionResponse audit(DecisionRequest request, Map<String, Object> features, String decision, int score,
            List<String> reasons) {
        auditRepo.save(request, features, decision, score, reasons);
        return new DecisionResponse(decision, score, reasons);
    }
}
