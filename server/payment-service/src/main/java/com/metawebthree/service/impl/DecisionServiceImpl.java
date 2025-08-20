package com.metawebthree.service.impl;

import com.metawebthree.dto.DecisionRequest;
import com.metawebthree.dto.DecisionResponse;
import com.metawebthree.entity.CreditProfile;
import com.metawebthree.repository.AuditRepo;
import com.metawebthree.repository.CreditProfileRepo;
import com.metawebthree.repository.FeatureRepo;
import com.metawebthree.repository.ModelScorer;
import com.metawebthree.repository.RuleRepo;
import com.metawebthree.service.DecisionService;
import com.metawebthree.service.entity.Rule;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import javax.script.ScriptEngineManager;
import java.time.LocalDateTime;
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
    private final CreditProfileRepo creditProfileRepo;

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
        // Check and adjust credit limit if approved
        if ("approve".equals(decision)) {
            adjustCreditLimit(request.getUserId(), score, features);
        }
        
        auditRepo.save(request, features, decision, score, reasons);
        return new DecisionResponse(decision, score, reasons);
    }

    private void adjustCreditLimit(String userId, int currentScore, Map<String, Object> features) {
        // Get current credit profile
        CreditProfile profile = creditProfileRepo.selectById(userId);
        if (profile == null) {
            profile = new CreditProfile();
            profile.setUserId(userId);
            creditProfileRepo.insert(profile);
        }

        // Check if eligible for adjustment (quarterly)
        if (profile.getLastLimitAdjustment() != null && 
            profile.getLastLimitAdjustment().isAfter(LocalDateTime.now().minusMonths(3))) {
            return;
        }

        // Check adjustment conditions
        boolean eligible = profile.getOverdueCount3m() == 0 &&
                         profile.getCreditUtilizationRate() > 60 &&
                         profile.getTransactionSuccessRate() > 95 &&
                         (currentScore - profile.getLastScore()) > 20;

        if (eligible) {
            // Calculate adjustment percentage (5-15%)
            double adjustmentPct = Math.min(
                0.05 + (currentScore - 600) * 0.001, // 600-700 score maps to 5-15%
                0.15
            );
            
            // Apply adjustment
            int newLimit = (int) (profile.getBaseCreditLimit() * (1 + adjustmentPct));
            profile.setCurrentCreditLimit(newLimit);
            profile.setLastLimitAdjustment(LocalDateTime.now());
            
            // Record adjustment
            List<Map<String, Object>> history = profile.getAdjustmentHistory() != null ? 
                profile.getAdjustmentHistory() : new ArrayList<>();
            history.add(Map.of(
                "date", LocalDateTime.now(),
                "oldLimit", profile.getCurrentCreditLimit(),
                "newLimit", newLimit,
                "adjustmentPct", adjustmentPct
            ));
            profile.setAdjustmentHistory(history);
            
            creditProfileRepo.save(profile);
        }
    }
}
