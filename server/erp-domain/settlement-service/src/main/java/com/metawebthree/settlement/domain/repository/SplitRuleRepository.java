package com.metawebthree.settlement.domain.repository;

import com.metawebthree.settlement.domain.entity.SplitRule;
import java.util.List;
import java.util.Optional;

public interface SplitRuleRepository {
    Optional<SplitRule> findById(Long id);
    Optional<SplitRule> findByRuleNo(String ruleNo);
    List<SplitRule> findByMerchantId(Long merchantId);
    List<SplitRule> findByStatus(SplitRule.SplitStatus status);
    List<SplitRule> findAll();
    void save(SplitRule rule);
    void update(SplitRule rule);
    void delete(Long id);
}