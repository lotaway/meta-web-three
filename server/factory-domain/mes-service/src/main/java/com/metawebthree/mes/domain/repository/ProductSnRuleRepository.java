package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.ProductSnRule;
import java.util.List;
import java.util.Optional;

public interface ProductSnRuleRepository {
    
    Optional<ProductSnRule> findById(Long id);
    
    Optional<ProductSnRule> findByProductId(Long productId);
    
    Optional<ProductSnRule> findByProductCode(String productCode);
    
    List<ProductSnRule> findAllActive();
    
    ProductSnRule save(ProductSnRule binding);
    
    void delete(Long id);
    
    boolean existsByProductId(Long productId);
}