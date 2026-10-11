package com.metawebthree.inventory.domain.repository;

import com.metawebthree.inventory.domain.entity.OutboundStrategy;
import java.util.List;

public interface OutboundStrategyRepository {
    OutboundStrategy findById(Long id);
    OutboundStrategy findByStrategyCode(String strategyCode);
    List<OutboundStrategy> findActiveByWarehouse(Long warehouseId);
    List<OutboundStrategy> findAllActive();
    OutboundStrategy save(OutboundStrategy strategy);
    boolean update(OutboundStrategy strategy);
    boolean delete(Long id);
}