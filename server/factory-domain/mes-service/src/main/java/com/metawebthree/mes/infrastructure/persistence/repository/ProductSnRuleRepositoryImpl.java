package com.metawebthree.mes.infrastructure.persistence.repository;

import com.metawebthree.mes.domain.entity.ProductSnRule;
import com.metawebthree.mes.domain.repository.ProductSnRuleRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ProductSnRuleDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.ProductSnRuleMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public class ProductSnRuleRepositoryImpl implements ProductSnRuleRepository {
    
    private final ProductSnRuleMapper mapper;
    
    public ProductSnRuleRepositoryImpl(ProductSnRuleMapper mapper) {
        this.mapper = mapper;
    }
    
    @Override
    public Optional<ProductSnRule> findById(Long id) {
        ProductSnRuleDO obj = mapper.selectById(id);
        return Optional.ofNullable(toEntity(obj));
    }
    
    @Override
    public Optional<ProductSnRule> findByProductId(Long productId) {
        ProductSnRuleDO obj = mapper.findActiveByProductId(productId);
        return Optional.ofNullable(toEntity(obj));
    }
    
    @Override
    public Optional<ProductSnRule> findByProductCode(String productCode) {
        ProductSnRuleDO obj = mapper.findActiveByProductCode(productCode);
        return Optional.ofNullable(toEntity(obj));
    }
    
    @Override
    public List<ProductSnRule> findAllActive() {
        return mapper.findAllActive().stream()
                .map(this::toEntity)
                .toList();
    }
    
    @Override
    public ProductSnRule save(ProductSnRule binding) {
        if (binding.getId() == null) {
            mapper.insert(toDO(binding));
        } else {
            mapper.updateById(toDO(binding));
        }
        return binding;
    }
    
    @Override
    public void delete(Long id) {
        mapper.deleteById(id);
    }
    
    @Override
    public boolean existsByProductId(Long productId) {
        return findByProductId(productId).isPresent();
    }
    
    private ProductSnRule toEntity(ProductSnRuleDO obj) {
        if (obj == null) return null;
        ProductSnRule entity = new ProductSnRule();
        entity.setId(obj.getId());
        entity.setProductId(obj.getProductId());
        entity.setProductCode(obj.getProductCode());
        entity.setCodeRuleId(obj.getCodeRuleId());
        entity.setRuleCode(obj.getRuleCode());
        entity.setIsActive(obj.getIsActive());
        entity.setDescription(obj.getDescription());
        entity.setCreatedAt(obj.getCreatedAt());
        entity.setUpdatedAt(obj.getUpdatedAt());
        return entity;
    }
    
    private ProductSnRuleDO toDO(ProductSnRule entity) {
        return ProductSnRuleDO.builder()
                .id(entity.getId())
                .productId(entity.getProductId())
                .productCode(entity.getProductCode())
                .codeRuleId(entity.getCodeRuleId())
                .ruleCode(entity.getRuleCode())
                .isActive(entity.getIsActive())
                .description(entity.getDescription())
                .createdAt(entity.getCreatedAt())
                .updatedAt(entity.getUpdatedAt())
                .build();
    }
}