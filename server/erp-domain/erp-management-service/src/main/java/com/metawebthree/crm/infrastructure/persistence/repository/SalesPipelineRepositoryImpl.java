package com.metawebthree.crm.infrastructure.persistence.repository;

import com.metawebthree.crm.domain.entity.SalesPipeline;
import com.metawebthree.crm.domain.repository.SalesPipelineRepository;
import com.metawebthree.crm.infrastructure.persistence.mapper.SalesPipelineMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public class SalesPipelineRepositoryImpl implements SalesPipelineRepository {

    private final SalesPipelineMapper mapper;

    public SalesPipelineRepositoryImpl(SalesPipelineMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public Optional<SalesPipeline> findById(Long id) {
        return Optional.ofNullable(mapper.selectById(id));
    }

    @Override
    public List<SalesPipeline> findAll() {
        return mapper.selectList(null);
    }

    @Override
    public SalesPipeline insert(SalesPipeline pipeline) {
        mapper.insert(pipeline);
        return pipeline;
    }

    @Override
    public SalesPipeline updateById(SalesPipeline pipeline) {
        mapper.updateById(pipeline);
        return pipeline;
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }
}
