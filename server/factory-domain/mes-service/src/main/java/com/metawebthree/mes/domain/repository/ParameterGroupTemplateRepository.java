package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.ParameterGroupTemplate;

import java.util.List;
import java.util.Optional;

/**
 * 参数组模板仓储接口
 */
public interface ParameterGroupTemplateRepository {
    
    Optional<ParameterGroupTemplate> findById(Long id);
    
    Optional<ParameterGroupTemplate> findByTemplateCode(String templateCode);
    
    List<ParameterGroupTemplate> findByProductType(String productType);
    
    List<ParameterGroupTemplate> findByStatus(ParameterGroupTemplate.TemplateStatus status);
    
    List<ParameterGroupTemplate> findByProductTypeAndStatus(String productType, ParameterGroupTemplate.TemplateStatus status);
    
    List<ParameterGroupTemplate> findAll();
    
    boolean existsByTemplateCode(String templateCode);
    
    boolean existsByProductType(String productType);
    
    long count();
    
    ParameterGroupTemplate save(ParameterGroupTemplate template);
    
    void deleteById(Long id);
    
    boolean existsById(Long id);
}