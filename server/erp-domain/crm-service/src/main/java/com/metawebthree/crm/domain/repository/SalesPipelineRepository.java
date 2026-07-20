package com.metawebthree.crm.domain.repository;

import com.metawebthree.crm.domain.entity.SalesPipeline;

import java.util.List;
import java.util.Optional;

public interface SalesPipelineRepository {
    Optional<SalesPipeline> findById(Long id);
    List<SalesPipeline> findAll();
    SalesPipeline insert(SalesPipeline pipeline);
    SalesPipeline updateById(SalesPipeline pipeline);
    void deleteById(Long id);
}
