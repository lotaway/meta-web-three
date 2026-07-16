package com.metawebthree.dom.domain.repository;

import com.metawebthree.dom.domain.entity.SourcingRule;
import java.util.List;
import java.util.Optional;

public interface SourcingRuleRepository {

    Optional<SourcingRule> findById(Long id);

    List<SourcingRule> findByRuleType(String ruleType);

    List<SourcingRule> findByRegion(String region);

    List<SourcingRule> findByEnabled(Boolean enabled);

    List<SourcingRule> findAll();

    SourcingRule save(SourcingRule rule);

    void delete(SourcingRule rule);
}
