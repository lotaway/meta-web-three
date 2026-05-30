package com.metawebthree.inventory.infrastructure.persistence.repository;

import com.metawebthree.inventory.domain.entity.OutboundStrategy;
import com.metawebthree.inventory.domain.repository.OutboundStrategyRepository;
import com.metawebthree.inventory.infrastructure.persistence.converter.OutboundStrategyConverter;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.OutboundStrategyDO;
import com.metawebthree.inventory.infrastructure.persistence.mapper.OutboundStrategyMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public class OutboundStrategyRepositoryImpl implements OutboundStrategyRepository {

    @Autowired
    private OutboundStrategyMapper mapper;

    @Autowired
    private OutboundStrategyConverter converter;

    @Override
    public OutboundStrategy findById(Long id) {
        OutboundStrategyDO dataObject = mapper.selectById(id);
        return converter.toEntity(dataObject);
    }

    @Override
    public OutboundStrategy findByStrategyCode(String strategyCode) {
        List<OutboundStrategyDO> list = mapper.selectList(
                new com.baomidou.mybatisplus.core.conditions.query.QueryWrapper<OutboundStrategyDO>()
                        .eq("strategy_code", strategyCode)
                        .last("LIMIT 1")
        );
        if (list == null || list.isEmpty()) {
            return null;
        }
        return converter.toEntity(list.get(0));
    }

    @Override
    public List<OutboundStrategy> findActiveByWarehouse(Long warehouseId) {
        List<OutboundStrategyDO> list = mapper.selectActiveByWarehouse(warehouseId);
        return converter.toEntityList(list);
    }

    @Override
    public List<OutboundStrategy> findAllActive() {
        List<OutboundStrategyDO> list = mapper.selectList(
                new com.baomidou.mybatisplus.core.conditions.query.QueryWrapper<OutboundStrategyDO>()
                        .eq("is_active", true)
                        .orderByAsc("priority")
        );
        return converter.toEntityList(list);
    }

    @Override
    public OutboundStrategy save(OutboundStrategy strategy) {
        strategy.setCreatedAt(LocalDateTime.now());
        strategy.setUpdatedAt(LocalDateTime.now());
        if (strategy.getVersion() == null) {
            strategy.setVersion(0);
        }
        mapper.insert(converter.toDataObject(strategy));
        return strategy;
    }

    @Override
    public boolean update(OutboundStrategy strategy) {
        strategy.setUpdatedAt(LocalDateTime.now());
        return mapper.updateById(converter.toDataObject(strategy)) > 0;
    }

    @Override
    public boolean delete(Long id) {
        return mapper.deleteById(id) > 0;
    }
}