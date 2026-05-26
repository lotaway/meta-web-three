package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.CodeRule;
import java.util.Optional;

/**
 * 编码规则仓储接口
 */
public interface CodeRuleRepository {
    
    /**
     * 根据ID查询
     */
    Optional<CodeRule> findById(Long id);
    
    /**
     * 根据规则编码查询
     */
    Optional<CodeRule> findByRuleCode(String ruleCode);
    
    /**
     * 根据业务类型查询启用的编码规则
     */
    Optional<CodeRule> findByBusinessTypeAndStatus(String businessType, CodeRule.RuleStatus status);
    
    /**
     * 保存编码规则
     */
    CodeRule save(CodeRule codeRule);
    
    /**
     * 删除编码规则
     */
    void delete(Long id);
    
    /**
     * 检查规则编码是否已存在
     */
    boolean existsByRuleCode(String ruleCode);
}