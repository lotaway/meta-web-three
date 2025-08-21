package com.metawebthree.service.impl;

import com.metawebthree.dto.DecisionRequest;
import com.metawebthree.dto.DecisionResponse;
import com.metawebthree.entity.CreditProfile;
import com.metawebthree.entity.Rule;
import com.metawebthree.enums.DecisionEnum;
import com.metawebthree.repository.AuditRepo;
import com.metawebthree.repository.CreditProfileRepo;
import com.metawebthree.repository.FeatureRepo;
import com.metawebthree.repository.RuleRepo;
import com.metawebthree.service.DecisionService;
import com.metawebthree.service.ModelScorerService;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import org.springframework.stereotype.Service;

import javax.script.ScriptEngine;
import javax.script.ScriptEngineManager;
import javax.script.ScriptException;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class DecisionServiceImpl implements DecisionService {

    private final RuleRepo ruleRepo;
    private final FeatureRepo featureRepo;
    private final ModelScorerService modelScorer;
    private final AuditRepo auditRepo;
    private final CreditProfileRepo creditProfileRepo;

    @Override
    public DecisionResponse decide(DecisionRequest request) {
        Map<String, Object> features = featureRepo.load(request);
        List<Rule> rules = ruleRepo.load(request.getScene());
        List<String> reasons = new ArrayList<>();

        for (Rule rule : rules) {
            if (eval(rule.getExpr(), features)) {
                if (DecisionEnum.REJECT.equals(rule.getAction())) {
                    return audit(request, features, DecisionEnum.REJECT, 0, Arrays.asList(rule.getCode()));
                }
                if (DecisionEnum.REVIEW.equals(rule.getAction())) {
                    reasons.add(rule.getCode());
                }
            }
        }

        int score = modelScorer.score(request.getScene(), features);
        if (score < 620) {
            return audit(request, features, DecisionEnum.REJECT, score, Arrays.asList("M402"));
        }

        DecisionEnum decision = reasons.isEmpty() ? DecisionEnum.APPROVE : DecisionEnum.REVIEW;
        return audit(request, features, decision, score, reasons);
    }

    private boolean eval(String expr, Map<String, Object> features) {
        try {
            ScriptEngineManager manager = new ScriptEngineManager();
            // Use rhino
            ScriptEngine engine = manager.getEngineByName("JavaScript");
            features.forEach(engine::put);
            Object result = engine.eval(expr);
            return result instanceof Boolean bReuslt && bReuslt;
        } catch (ScriptException e) {
            e.printStackTrace();
            log.error("Error in eval rule", e);
            return false;
        }
    }

    private DecisionResponse audit(DecisionRequest request, Map<String, Object> features, DecisionEnum decision,
            int score,
            List<String> reasons) {
        // Check and adjust credit limit if approved
        if (DecisionEnum.APPROVE.equals(decision)) {
            adjustCreditLimit(request.getUserId(), score, features);
        }

        auditRepo.save(request, features, decision, score, reasons);
        return new DecisionResponse(decision, score, reasons);
    }

    private void adjustCreditLimit(Long userId, int currentScore, Map<String, Object> features) {
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
                    0.15);

            // Apply adjustment
            int newLimit = (int) (profile.getBaseCreditLimit() * (1 + adjustmentPct));
            profile.setCurrentCreditLimit(newLimit);
            profile.setLastLimitAdjustment(LocalDateTime.now());

            // Record adjustment
            List<Map<String, Object>> history = profile.getAdjustmentHistory() != null ? profile.getAdjustmentHistory()
                    : new ArrayList<>();
            history.add(Map.of(
                    "date", LocalDateTime.now(),
                    "oldLimit", profile.getCurrentCreditLimit(),
                    "newLimit", newLimit,
                    "adjustmentPct", adjustmentPct));
            profile.setAdjustmentHistory(history);
            creditProfileRepo.insert(profile);
        }
    }
}
