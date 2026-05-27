package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.MaterialIssueConfig;

import java.util.List;
import java.util.Optional;

public interface MaterialIssueConfigRepository {
    
    Optional<MaterialIssueConfig> findById(Long id);
    
    MaterialIssueConfig save(MaterialIssueConfig config);
    
    void deleteById(Long id);
    
    Optional<MaterialIssueConfig> findActiveByWorkshopAndProduct(String workshopId, String productCode);
    
    Optional<MaterialIssueConfig> findActiveByWorkshopDefault(String workshopId);
    
    List<MaterialIssueConfig> findAllActive();
    
    Optional<MaterialIssueConfig> findActiveByConfigCode(String configCode);
    
    List<MaterialIssueConfig> findActiveByIssueMode(String issueMode);
}