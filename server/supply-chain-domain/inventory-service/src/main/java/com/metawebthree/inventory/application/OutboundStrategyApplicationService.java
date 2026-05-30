package com.metawebthree.inventory.application;

import com.metawebthree.inventory.application.dto.BatchAllocationDTO;
import com.metawebthree.inventory.application.dto.OutboundStrategyDTO;
import java.util.List;

public interface OutboundStrategyApplicationService {

    BatchAllocationDTO allocateBatches(String skuCode, Long warehouseId, Integer quantity, String strategyType);

    BatchAllocationDTO allocateBatchesByStrategyId(Long strategyId, String skuCode, Long warehouseId, Integer quantity);

    OutboundStrategyDTO createStrategy(OutboundStrategyDTO dto);

    OutboundStrategyDTO updateStrategy(OutboundStrategyDTO dto);

    boolean deleteStrategy(Long id);

    OutboundStrategyDTO getStrategyById(Long id);

    List<OutboundStrategyDTO> listStrategies(Long warehouseId, Boolean isActive);

    OutboundStrategyDTO getEffectiveStrategy(String skuCode, Long warehouseId);
}