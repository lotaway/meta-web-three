package com.metawebthree.inventory.application;

import com.metawebthree.inventory.application.dto.BatchAllocationDTO;
import com.metawebthree.inventory.application.dto.OutboundStrategyDTO;
import com.metawebthree.inventory.domain.entity.InventoryBatch;
import com.metawebthree.inventory.domain.entity.OutboundStrategy;
import com.metawebthree.inventory.domain.repository.InventoryBatchRepository;
import com.metawebthree.inventory.domain.repository.OutboundStrategyRepository;
import com.metawebthree.inventory.domain.service.OutboundStrategyDomainService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class OutboundStrategyApplicationServiceImpl implements OutboundStrategyApplicationService {

    @Autowired
    private InventoryBatchRepository batchRepository;

    @Autowired
    private OutboundStrategyRepository strategyRepository;

    @Autowired
    private OutboundStrategyDomainService strategyDomainService;

    @Override
    @Transactional
    public BatchAllocationDTO allocateBatches(String skuCode, Long warehouseId, Integer quantity, String strategyType) {
        List<InventoryBatch> availableBatches = batchRepository.findAvailableBatches(skuCode, warehouseId);
        
        List<InventoryBatch> sortedBatches = strategyDomainService.selectBatchesByStrategy(
                availableBatches, strategyType, null);
        
        List<InventoryBatch> allocated = strategyDomainService.allocateQuantity(sortedBatches, quantity);
        
        return buildAllocationDTO(skuCode, warehouseId, quantity, strategyType, allocated);
    }

    @Override
    @Transactional
    public BatchAllocationDTO allocateBatchesByStrategyId(Long strategyId, String skuCode, Long warehouseId, Integer quantity) {
        OutboundStrategy strategy = strategyRepository.findById(strategyId);
        if (strategy == null) {
            throw new IllegalArgumentException("Strategy not found: " + strategyId);
        }

        List<InventoryBatch> availableBatches = batchRepository.findAvailableBatches(skuCode, warehouseId);
        
        List<InventoryBatch> sortedBatches = strategyDomainService.selectBatchesByStrategy(
                availableBatches, strategy.getStrategyType(), strategy.getSpecificBatchNo());
        
        List<InventoryBatch> allocated = strategyDomainService.allocateQuantity(sortedBatches, quantity);
        
        return buildAllocationDTO(skuCode, warehouseId, quantity, strategy.getStrategyType(), allocated);
    }

    @Override
    @Transactional
    public OutboundStrategyDTO createStrategy(OutboundStrategyDTO dto) {
        OutboundStrategy entity = toEntity(dto);
        entity.setIsActive(true);
        OutboundStrategy saved = strategyRepository.save(entity);
        return toDTO(saved);
    }

    @Override
    @Transactional
    public OutboundStrategyDTO updateStrategy(OutboundStrategyDTO dto) {
        OutboundStrategy existing = strategyRepository.findById(dto.getId());
        if (existing == null) {
            throw new IllegalArgumentException("Strategy not found: " + dto.getId());
        }
        existing.setStrategyName(dto.getStrategyName());
        existing.setStrategyType(dto.getStrategyType());
        existing.setWarehouseId(dto.getWarehouseId());
        existing.setWarehouseCode(dto.getWarehouseCode());
        existing.setSkuCode(dto.getSkuCode());
        existing.setSkuCodePattern(dto.getSkuCodePattern());
        existing.setPriority(dto.getPriority());
        existing.setSpecificBatchNo(dto.getSpecificBatchNo());
        existing.setIsActive(dto.getIsActive());
        existing.setRemark(dto.getRemark());
        
        strategyRepository.update(existing);
        return toDTO(existing);
    }

    @Override
    @Transactional
    public boolean deleteStrategy(Long id) {
        return strategyRepository.delete(id);
    }

    @Override
    public OutboundStrategyDTO getStrategyById(Long id) {
        OutboundStrategy entity = strategyRepository.findById(id);
        return entity != null ? toDTO(entity) : null;
    }

    @Override
    public List<OutboundStrategyDTO> listStrategies(Long warehouseId, Boolean isActive) {
        List<OutboundStrategy> strategies;
        if (warehouseId != null) {
            strategies = strategyRepository.findActiveByWarehouse(warehouseId);
        } else {
            strategies = strategyRepository.findAllActive();
        }
        
        if (isActive != null) {
            strategies = strategies.stream()
                    .filter(s -> isActive.equals(s.getIsActive()))
                    .collect(Collectors.toList());
        }
        
        return strategies.stream()
                .map(this::toDTO)
                .collect(Collectors.toList());
    }

    @Override
    public OutboundStrategyDTO getEffectiveStrategy(String skuCode, Long warehouseId) {
        List<OutboundStrategy> strategies = strategyRepository.findActiveByWarehouse(warehouseId);
        
        OutboundStrategy matched = strategies.stream()
                .filter(s -> s.matchesWarehouse(warehouseId))
                .filter(s -> s.matchesSku(skuCode))
                .findFirst()
                .orElse(null);
        
        if (matched == null) {
            strategies = strategyRepository.findAllActive();
            matched = strategies.stream()
                    .filter(s -> s.matchesWarehouse(null))
                    .filter(s -> s.matchesSku(skuCode))
                    .findFirst()
                    .orElse(null);
        }
        
        if (matched == null) {
            matched = new OutboundStrategy();
            matched.setStrategyType("FIFO");
        }
        
        return toDTO(matched);
    }

    private BatchAllocationDTO buildAllocationDTO(String skuCode, Long warehouseId, Integer quantity, 
                                                    String strategyType, List<InventoryBatch> allocated) {
        BatchAllocationDTO dto = new BatchAllocationDTO();
        dto.setSkuCode(skuCode);
        dto.setWarehouseId(warehouseId);
        dto.setTotalRequiredQuantity(quantity);
        dto.setTotalAllocatedQuantity(allocated.stream().mapToInt(InventoryBatch::getQuantity).sum());
        dto.setStrategyType(strategyType != null ? strategyType : "FIFO");
        
        List<BatchAllocationDTO.BatchPickDetail> details = allocated.stream()
                .map(batch -> {
                    BatchAllocationDTO.BatchPickDetail detail = new BatchAllocationDTO.BatchPickDetail();
                    detail.setBatchId(batch.getId());
                    detail.setBatchNo(batch.getBatchNo());
                    detail.setAllocatedQuantity(batch.getQuantity());
                    detail.setInboundDate(batch.getInboundDate());
                    detail.setExpiryDate(batch.getExpiryDate());
                    detail.setLocationCode(batch.getLocationCode());
                    detail.setUnitCost(batch.getUnitCost());
                    return detail;
                })
                .collect(Collectors.toList());
        
        dto.setBatches(details);
        return dto;
    }

    private OutboundStrategy toEntity(OutboundStrategyDTO dto) {
        OutboundStrategy entity = new OutboundStrategy();
        entity.setId(dto.getId());
        entity.setStrategyCode(dto.getStrategyCode());
        entity.setStrategyName(dto.getStrategyName());
        entity.setStrategyType(dto.getStrategyType());
        entity.setWarehouseId(dto.getWarehouseId());
        entity.setWarehouseCode(dto.getWarehouseCode());
        entity.setSkuCode(dto.getSkuCode());
        entity.setSkuCodePattern(dto.getSkuCodePattern());
        entity.setPriority(dto.getPriority());
        entity.setSpecificBatchNo(dto.getSpecificBatchNo());
        entity.setIsActive(dto.getIsActive());
        entity.setRemark(dto.getRemark());
        entity.setCreator(dto.getCreator());
        return entity;
    }

    private OutboundStrategyDTO toDTO(OutboundStrategy entity) {
        OutboundStrategyDTO dto = new OutboundStrategyDTO();
        dto.setId(entity.getId());
        dto.setStrategyCode(entity.getStrategyCode());
        dto.setStrategyName(entity.getStrategyName());
        dto.setStrategyType(entity.getStrategyType());
        dto.setWarehouseId(entity.getWarehouseId());
        dto.setWarehouseCode(entity.getWarehouseCode());
        dto.setSkuCode(entity.getSkuCode());
        dto.setSkuCodePattern(entity.getSkuCodePattern());
        dto.setPriority(entity.getPriority());
        dto.setSpecificBatchNo(entity.getSpecificBatchNo());
        dto.setIsActive(entity.getIsActive());
        dto.setRemark(entity.getRemark());
        dto.setCreator(entity.getCreator());
        dto.setCreatedAt(entity.getCreatedAt());
        dto.setUpdatedAt(entity.getUpdatedAt());
        return dto;
    }
}