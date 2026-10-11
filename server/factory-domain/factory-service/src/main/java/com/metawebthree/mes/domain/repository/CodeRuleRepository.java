package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.CodeRule;
import java.util.List;
import java.util.Optional;

public interface CodeRuleRepository {
    
    Optional<CodeRule> findById(Long id);
    
    Optional<CodeRule> findByRuleCode(String ruleCode);
    
    Optional<CodeRule> findByBusinessTypeAndStatus(String businessType, CodeRule.RuleStatus status);
    
    List<CodeRule> findAllActive();
    
    CodeRule save(CodeRule codeRule);
    
    void delete(Long id);
    
    boolean existsByRuleCode(String ruleCode);
}