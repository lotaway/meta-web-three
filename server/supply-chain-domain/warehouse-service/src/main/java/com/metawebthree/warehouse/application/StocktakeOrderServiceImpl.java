package com.metawebthree.warehouse.application;

import com.metawebthree.warehouse.application.dto.StocktakeOrderDTO;
import com.metawebthree.warehouse.application.dto.StocktakeOrderItemDTO;
import com.metawebthree.warehouse.domain.entity.StocktakeOrder;
import com.metawebthree.warehouse.domain.entity.StocktakeOrderItem;
import com.metawebthree.warehouse.domain.repository.StocktakeOrderRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
public class StocktakeOrderServiceImpl implements StocktakeOrderService {

    private final StocktakeOrderRepository stocktakeOrderRepository;

    public StocktakeOrderServiceImpl(StocktakeOrderRepository stocktakeOrderRepository) {
        this.stocktakeOrderRepository = stocktakeOrderRepository;
    }

    @Override
    @Transactional
    public StocktakeOrderDTO createStocktakeOrder(StocktakeOrderDTO dto) {
        StocktakeOrder order = new StocktakeOrder();
        order.setOrderNo("ST" + UUID.randomUUID().toString().substring(0, 8).toUpperCase());
        order.setType(dto.getType());
        order.setWarehouseId(dto.getWarehouseId());
        order.setWarehouseName(dto.getWarehouseName());
        order.setLocationId(dto.getLocationId());
        order.setLocationName(dto.getLocationName());
        order.setStatus(StocktakeOrder.STATUS_DRAFT);
        order.setOperator(dto.getOperator());
        order.setPlannedDate(dto.getPlannedDate());
        order.setRemark(dto.getRemark());
        order.setCreatedBy(dto.getCreatedBy());
        order.setCreatedAt(LocalDateTime.now());
        order.setUpdatedAt(LocalDateTime.now());
        
        if (dto.getItems() != null && !dto.getItems().isEmpty()) {
            List<StocktakeOrderItem> items = dto.getItems().stream()
                    .map(this::convertToEntityItem)
                    .collect(Collectors.toList());
            order.setItems(items);
            order.setTotalSkuCount(items.size());
        }
        
        stocktakeOrderRepository.insert(order);
        return convertToDTO(order);
    }

    @Override
    @Transactional
    public StocktakeOrderDTO submitStocktakeOrder(String orderNo) {
        StocktakeOrder order = stocktakeOrderRepository.findByOrderNo(orderNo)
                .orElseThrow(() -> new RuntimeException("Stocktake order not found: " + orderNo));
        order.submit();
        order.setUpdatedAt(LocalDateTime.now());
        stocktakeOrderRepository.update(order);
        return convertToDTO(order);
    }

    @Override
    @Transactional
    public StocktakeOrderDTO startStocktake(String orderNo) {
        StocktakeOrder order = stocktakeOrderRepository.findByOrderNo(orderNo)
                .orElseThrow(() -> new RuntimeException("Stocktake order not found: " + orderNo));
        order.start();
        order.setUpdatedAt(LocalDateTime.now());
        stocktakeOrderRepository.update(order);
        return convertToDTO(order);
    }

    @Override
    @Transactional
    public StocktakeOrderDTO completeCounting(String orderNo) {
        StocktakeOrder order = stocktakeOrderRepository.findByOrderNo(orderNo)
                .orElseThrow(() -> new RuntimeException("Stocktake order not found: " + orderNo));
        order.completeCounting();
        order.setUpdatedAt(LocalDateTime.now());
        stocktakeOrderRepository.update(order);
        return convertToDTO(order);
    }

    @Override
    @Transactional
    public StocktakeOrderDTO reportDiscrepancy(String orderNo) {
        StocktakeOrder order = stocktakeOrderRepository.findByOrderNo(orderNo)
                .orElseThrow(() -> new RuntimeException("Stocktake order not found: " + orderNo));
        order.reportDiscrepancy();
        
        if (order.getItems() != null) {
            int discrepancyCount = 0;
            for (StocktakeOrderItem item : order.getItems()) {
                if (item.getDiscrepancyQuantity() != null && 
                    item.getDiscrepancyQuantity().compareTo(java.math.BigDecimal.ZERO) != 0) {
                    discrepancyCount++;
                }
            }
            order.setDiscrepancyCount(discrepancyCount);
        }
        
        order.setUpdatedAt(LocalDateTime.now());
        stocktakeOrderRepository.update(order);
        return convertToDTO(order);
    }

    @Override
    @Transactional
    public StocktakeOrderDTO adjustInventory(String orderNo) {
        StocktakeOrder order = stocktakeOrderRepository.findByOrderNo(orderNo)
                .orElseThrow(() -> new RuntimeException("Stocktake order not found: " + orderNo));
        order.adjustInventory();
        order.setUpdatedAt(LocalDateTime.now());
        stocktakeOrderRepository.update(order);
        return convertToDTO(order);
    }

    @Override
    @Transactional
    public StocktakeOrderDTO completeStocktake(String orderNo) {
        StocktakeOrder order = stocktakeOrderRepository.findByOrderNo(orderNo)
                .orElseThrow(() -> new RuntimeException("Stocktake order not found: " + orderNo));
        order.complete();
        order.setUpdatedAt(LocalDateTime.now());
        stocktakeOrderRepository.update(order);
        return convertToDTO(order);
    }

    @Override
    @Transactional
    public StocktakeOrderDTO cancelStocktake(String orderNo) {
        StocktakeOrder order = stocktakeOrderRepository.findByOrderNo(orderNo)
                .orElseThrow(() -> new RuntimeException("Stocktake order not found: " + orderNo));
        order.cancel();
        order.setUpdatedAt(LocalDateTime.now());
        stocktakeOrderRepository.update(order);
        return convertToDTO(order);
    }

    @Override
    public StocktakeOrderDTO queryStocktakeOrder(String orderNo) {
        StocktakeOrder order = stocktakeOrderRepository.findByOrderNo(orderNo)
                .orElseThrow(() -> new RuntimeException("Stocktake order not found: " + orderNo));
        return convertToDTO(order);
    }

    @Override
    public List<StocktakeOrderDTO> listStocktakeOrders(Long warehouseId, String status) {
        List<StocktakeOrder> orders;
        if (warehouseId != null && status != null) {
            orders = stocktakeOrderRepository.findByWarehouseIdAndStatus(warehouseId, status);
        } else if (warehouseId != null) {
            orders = stocktakeOrderRepository.findByWarehouseId(warehouseId);
        } else if (status != null) {
            orders = stocktakeOrderRepository.findByStatus(status);
        } else {
            orders = stocktakeOrderRepository.findByWarehouseId(null);
        }
        return orders.stream().map(this::convertToDTO).collect(Collectors.toList());
    }

    private StocktakeOrderDTO convertToDTO(StocktakeOrder order) {
        if (order == null) {
            return null;
        }
        StocktakeOrderDTO dto = new StocktakeOrderDTO();
        dto.setId(order.getId());
        dto.setOrderNo(order.getOrderNo());
        dto.setType(order.getType());
        dto.setWarehouseId(order.getWarehouseId());
        dto.setWarehouseName(order.getWarehouseName());
        dto.setLocationId(order.getLocationId());
        dto.setLocationName(order.getLocationName());
        dto.setStatus(order.getStatus());
        dto.setOperator(order.getOperator());
        dto.setPlannedDate(order.getPlannedDate());
        dto.setStartDate(order.getStartDate());
        dto.setEndDate(order.getEndDate());
        dto.setTotalSkuCount(order.getTotalSkuCount());
        dto.setCheckedSkuCount(order.getCheckedSkuCount());
        dto.setDiscrepancyCount(order.getDiscrepancyCount());
        dto.setTotalDiscrepancyAmount(order.getTotalDiscrepancyAmount());
        dto.setRemark(order.getRemark());
        dto.setCreatedBy(order.getCreatedBy());
        dto.setCreatedAt(order.getCreatedAt());
        
        if (order.getItems() != null) {
            dto.setItems(order.getItems().stream()
                    .map(this::convertToDTOItem)
                    .collect(Collectors.toList()));
        }
        
        return dto;
    }

    private StocktakeOrderItemDTO convertToDTOItem(StocktakeOrderItem item) {
        if (item == null) {
            return null;
        }
        StocktakeOrderItemDTO dto = new StocktakeOrderItemDTO();
        dto.setId(item.getId());
        dto.setStocktakeOrderId(item.getStocktakeOrderId());
        dto.setSkuCode(item.getSkuCode());
        dto.setSkuName(item.getSkuName());
        dto.setUnit(item.getUnit());
        dto.setSystemQuantity(item.getSystemQuantity());
        dto.setCountedQuantity(item.getCountedQuantity());
        dto.setDiscrepancyQuantity(item.getDiscrepancyQuantity());
        dto.setDiscrepancyAmount(item.getDiscrepancyAmount());
        dto.setDiscrepancyReason(item.getDiscrepancyReason());
        dto.setStatus(item.getStatus());
        dto.setCounter(item.getCounter());
        dto.setCountedAt(item.getCountedAt());
        dto.setChecker(item.getChecker());
        dto.setCheckedAt(item.getCheckedAt());
        dto.setAdjuster(item.getAdjuster());
        dto.setAdjustedAt(item.getAdjustedAt());
        dto.setRemark(item.getRemark());
        return dto;
    }

    private StocktakeOrderItem convertToEntityItem(StocktakeOrderItemDTO dto) {
        if (dto == null) {
            return null;
        }
        StocktakeOrderItem item = new StocktakeOrderItem();
        item.setSkuCode(dto.getSkuCode());
        item.setSkuName(dto.getSkuName());
        item.setUnit(dto.getUnit());
        item.setSystemQuantity(dto.getSystemQuantity());
        item.setStatus(StocktakeOrderItem.STATUS_PENDING);
        item.setCreatedAt(LocalDateTime.now());
        item.setUpdatedAt(LocalDateTime.now());
        return item;
    }
}
