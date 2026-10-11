package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.QcTriggerRule;
import java.util.List;
import java.util.Optional;

public interface QcTriggerRuleRepository {
    Optional<QcTriggerRule> findById(Long id);
    Optional<QcTriggerRule> findByRuleCode(String ruleCode);
    List<QcTriggerRule> findAll();
    List<QcTriggerRule> findByTriggerType(QcTriggerRule.TriggerType triggerType);
    List<QcTriggerRule> findByIsEnabled(Boolean isEnabled);
    QcTriggerRule save(QcTriggerRule triggerRule);
    void update(QcTriggerRule triggerRule);
    void deleteById(Long id);
    Boolean existsByRuleCode(String ruleCode);
}