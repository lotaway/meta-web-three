package com.metawebthree.warehouse.application;

import com.metawebthree.warehouse.application.dto.WarehouseDTO;
import com.metawebthree.warehouse.application.dto.InboundOrderDTO;
import com.metawebthree.warehouse.application.dto.InboundOrderItemDTO;
import com.metawebthree.warehouse.domain.entity.Warehouse;
import com.metawebthree.warehouse.domain.entity.InboundOrder;
import com.metawebthree.warehouse.domain.entity.InboundOrderItem;
import com.metawebthree.warehouse.infrastructure.event.WarehouseEventPublisher;
import com.metawebthree.warehouse.infrastructure.persistence.repository.WarehouseRepository;
import com.metawebthree.warehouse.infrastructure.persistence.repository.InboundOrderRepository;
import org.springframework.stereotype.Service;
import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
public class WarehouseApplicationServiceImpl implements WarehouseApplicationService {

    private final WarehouseRepository warehouseRepository;
    private final InboundOrderRepository inboundOrderRepository;
    private final WarehouseEventPublisher eventPublisher;

    public WarehouseApplicationServiceImpl(WarehouseRepository warehouseRepository,
                                           InboundOrderRepository inboundOrderRepository,
                                           WarehouseEventPublisher eventPublisher) {
        this.warehouseRepository = warehouseRepository;
        this.inboundOrderRepository = inboundOrderRepository;
        this.eventPublisher = eventPublisher;
    }

    @Override
    public WarehouseDTO createWarehouse(WarehouseDTO dto) {
        Warehouse warehouse = new Warehouse();
        warehouse.setWarehouseCode(generateWarehouseCode());
        warehouse.setWarehouseName(dto.getWarehouseName());
        warehouse.setWarehouseType(dto.getWarehouseType());
        warehouse.setProvince(dto.getProvince());
        warehouse.setCity(dto.getCity());
        warehouse.setDistrict(dto.getDistrict());
        warehouse.setAddress(dto.getAddress());
        warehouse.setContact(dto.getContact());
        warehouse.setPhone(dto.getPhone());
        warehouse.setTotalCapacity(dto.getTotalCapacity());
        warehouse.setUsedCapacity(0);
        warehouse.setAvailableCapacity(dto.getTotalCapacity());
        warehouse.setStatus("ACTIVE");
        warehouse.setCreatedAt(LocalDateTime.now());
        warehouse.setUpdatedAt(LocalDateTime.now());
        
        Warehouse saved = warehouseRepository.save(warehouse);
        eventPublisher.publishCreated(saved.getId(), saved.getWarehouseCode(), saved.getWarehouseName());
        
        return toWarehouseDTO(saved);
    }

    @Override
    public WarehouseDTO updateWarehouse(Long id, WarehouseDTO dto) {
        return warehouseRepository.findById(id)
            .map(warehouse -> {
                if (dto.getWarehouseName() != null) {
                    warehouse.setWarehouseName(dto.getWarehouseName());
                }
                if (dto.getContact() != null) {
                    warehouse.setContact(dto.getContact());
                }
                if (dto.getPhone() != null) {
                    warehouse.setPhone(dto.getPhone());
                }
                if (dto.getAddress() != null) {
                    warehouse.setAddress(dto.getAddress());
                }
                warehouse.setUpdatedAt(LocalDateTime.now());
                return toWarehouseDTO(warehouseRepository.save(warehouse));
            })
            .orElse(null);
    }

    @Override
    public WarehouseDTO queryWarehouse(Long id) {
        return warehouseRepository.findById(id)
            .map(this::toWarehouseDTO)
            .orElse(null);
    }

    @Override
    public List<WarehouseDTO> listWarehouses(String status) {
        // 简化实现，返回空列表或通过其他方式实现
        return List.of();
    }

    @Override
    public InboundOrderDTO createInboundOrder(InboundOrderDTO dto) {
        InboundOrder order = new InboundOrder();
        order.setOrderNo(generateInboundOrderNo());
        order.setInboundType(dto.getInboundType());
        order.setWarehouseId(dto.getWarehouseId());
        order.setSupplierCode(dto.getSupplierCode());
        order.setStatus("PENDING");
        order.setRemark(dto.getRemark());
        order.setOperator(dto.getOperator());
        order.setPlanArrivalTime(dto.getPlanArrivalTime());
        order.setCreatedAt(LocalDateTime.now());
        order.setUpdatedAt(LocalDateTime.now());
        
        if (dto.getItems() != null) {
            List<InboundOrderItem> items = dto.getItems().stream()
                .map(itemDTO -> {
                    InboundOrderItem item = new InboundOrderItem();
                    item.setSkuCode(itemDTO.getSkuCode());
                    item.setProductName(itemDTO.getProductName());
                    item.setPlanQuantity(itemDTO.getPlanQuantity());
                    item.setActualQuantity(itemDTO.getActualQuantity());
                    item.setUnitCost(itemDTO.getUnitCost());
                    item.setBatchNo(itemDTO.getBatchNo());
                    item.setStatus("PENDING");
                    return item;
                })
                .collect(Collectors.toList());
            order.setItems(items);
        }
        
        InboundOrder saved = inboundOrderRepository.save(order);
        eventPublisher.publishInboundOrderCreated(saved.getOrderNo(), saved.getWarehouseId());
        
        return toInboundOrderDTO(saved);
    }

    @Override
    public InboundOrderDTO confirmInboundOrder(String orderNo) {
        return inboundOrderRepository.findByOrderNo(orderNo)
            .map(order -> {
                order.confirm();
                order.setUpdatedAt(LocalDateTime.now());
                return toInboundOrderDTO(inboundOrderRepository.save(order));
            })
            .orElse(null);
    }

    @Override
    public InboundOrderDTO completeInboundOrder(String orderNo, InboundOrderDTO dto) {
        return inboundOrderRepository.findByOrderNo(orderNo)
            .map(order -> {
                order.complete();
                order.setActualArrivalTime(dto.getActualArrivalTime() != null 
                    ? dto.getActualArrivalTime() 
                    : LocalDateTime.now());
                order.setUpdatedAt(LocalDateTime.now());
                InboundOrder saved = inboundOrderRepository.save(order);
                
                // 发布入库完成事件和入库明细事件
                eventPublisher.publishInboundOrderCompleted(orderNo);
                if (order.getItems() != null) {
                    for (InboundOrderItem item : order.getItems()) {
                        int qty = item.getActualQuantity() != null ? item.getActualQuantity() : (item.getPlanQuantity() != null ? item.getPlanQuantity() : 0);
                        eventPublisher.publishStockIn(
                            order.getWarehouseId(),
                            item.getSkuCode(),
                            qty,
                            orderNo
                        );
                    }
                }
                
                return toInboundOrderDTO(saved);
            })
            .orElse(null);
    }

    @Override
    public InboundOrderDTO queryInboundOrder(String orderNo) {
        return inboundOrderRepository.findByOrderNo(orderNo)
            .map(this::toInboundOrderDTO)
            .orElse(null);
    }

    @Override
    public List<InboundOrderDTO> listInboundOrders(Long warehouseId, String status) {
        List<InboundOrder> orders;
        if (warehouseId != null) {
            orders = inboundOrderRepository.findByWarehouseId(warehouseId);
        } else if (status != null && !status.isEmpty()) {
            orders = inboundOrderRepository.findByStatus(status);
        } else {
            orders = List.of();
        }
        return orders.stream()
            .map(this::toInboundOrderDTO)
            .collect(Collectors.toList());
    }

    private String generateWarehouseCode() {
        return "WH" + System.currentTimeMillis() + UUID.randomUUID().toString().substring(0, 4).toUpperCase();
    }

    private String generateInboundOrderNo() {
        return "IB" + System.currentTimeMillis() + UUID.randomUUID().toString().substring(0, 4).toUpperCase();
    }

    private WarehouseDTO toWarehouseDTO(Warehouse warehouse) {
        WarehouseDTO dto = new WarehouseDTO();
        dto.setId(warehouse.getId());
        dto.setWarehouseCode(warehouse.getWarehouseCode());
        dto.setWarehouseName(warehouse.getWarehouseName());
        dto.setWarehouseType(warehouse.getWarehouseType());
        dto.setProvince(warehouse.getProvince());
        dto.setCity(warehouse.getCity());
        dto.setDistrict(warehouse.getDistrict());
        dto.setAddress(warehouse.getAddress());
        dto.setContact(warehouse.getContact());
        dto.setPhone(warehouse.getPhone());
        dto.setTotalCapacity(warehouse.getTotalCapacity());
        dto.setUsedCapacity(warehouse.getUsedCapacity());
        dto.setAvailableCapacity(warehouse.getAvailableCapacity());
        dto.setStatus(warehouse.getStatus());
        dto.setCreatedAt(warehouse.getCreatedAt());
        return dto;
    }

    private InboundOrderDTO toInboundOrderDTO(InboundOrder order) {
        InboundOrderDTO dto = new InboundOrderDTO();
        dto.setId(order.getId());
        dto.setOrderNo(order.getOrderNo());
        dto.setInboundType(order.getInboundType());
        dto.setWarehouseId(order.getWarehouseId());
        dto.setSupplierCode(order.getSupplierCode());
        dto.setStatus(order.getStatus());
        dto.setRemark(order.getRemark());
        dto.setOperator(order.getOperator());
        dto.setPlanArrivalTime(order.getPlanArrivalTime());
        dto.setActualArrivalTime(order.getActualArrivalTime());
        dto.setCompletedAt(order.getCompletedAt());
        dto.setCreatedAt(order.getCreatedAt());
        
        if (order.getItems() != null) {
            List<InboundOrderItemDTO> items = order.getItems().stream()
                .map(this::toInboundOrderItemDTO)
                .collect(Collectors.toList());
            dto.setItems(items);
        }
        
        return dto;
    }

    private InboundOrderItemDTO toInboundOrderItemDTO(InboundOrderItem item) {
        InboundOrderItemDTO dto = new InboundOrderItemDTO();
        dto.setId(item.getId());
        dto.setOrderId(item.getOrderId());
        dto.setSkuCode(item.getSkuCode());
        dto.setProductName(item.getProductName());
        dto.setPlanQuantity(item.getPlanQuantity());
        dto.setActualQuantity(item.getActualQuantity());
        dto.setLocationId(item.getLocationId());
        dto.setStatus(item.getStatus());
        dto.setUnitCost(item.getUnitCost());
        dto.setBatchNo(item.getBatchNo());
        dto.setProductionDate(item.getProductionDate());
        dto.setExpiryDate(item.getExpiryDate());
        return dto;
    }
}