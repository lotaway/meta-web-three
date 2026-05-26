package com.metawebthree.mes.infrastructure.persistence.repository;

import com.metawebthree.mes.domain.entity.CodeRule;
import com.metawebthree.mes.domain.repository.CodeRuleRepository;
import org.springframework.stereotype.Repository;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Repository
public class CodeRuleRepositoryImpl implements CodeRuleRepository {
    
    private final Map<Long, CodeRule> storage = new ConcurrentHashMap<>();
    private final AtomicLong idGen = new AtomicLong(1);
    
    @Override
    public Optional<CodeRule> findById(Long id) {
        return Optional.ofNullable(storage.get(id));
    }
    
    @Override
    public Optional<CodeRule> findByRuleCode(String ruleCode) {
        return storage.values().stream()
                .filter(r -> r.getRuleCode().equals(ruleCode))
                .findFirst();
    }
    
    @Override
    public Optional<CodeRule> findByBusinessTypeAndStatus(String businessType, CodeRule.RuleStatus status) {
        return storage.values().stream()
                .filter(r -> r.getBusinessType().equals(businessType))
                .filter(r -> r.getStatus() == status)
                .findFirst();
    }
    
    @Override
    public CodeRule save(CodeRule codeRule) {
        if (codeRule.getId() == null) {
            codeRule.setId(idGen.getAndIncrement());
        }
        storage.put(codeRule.getId(), codeRule);
        return codeRule;
    }
    
    @Override
    public void delete(Long id) {
        storage.remove(id);
    }
    
    @Override
    public boolean existsByRuleCode(String ruleCode) {
        return storage.values().stream()
                .anyMatch(r -> r.getRuleCode().equals(ruleCode));
    }
}