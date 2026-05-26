package com.metawebthree.warehouse.application;

import com.metawebthree.warehouse.application.dto.WarehouseDTO;
import com.metawebthree.warehouse.application.dto.InboundOrderDTO;
import com.metawebthree.warehouse.application.dto.InboundOrderItemDTO;
import com.metawebthree.warehouse.domain.entity.Warehouse;
import com.metawebthree.warehouse.domain.entity.InboundOrder;
import com.metawebthree.warehouse.domain.entity.InboundOrderItem;
import com.metawebthree.warehouse.infrastructure.event.WarehouseDomainEventPublisher;
import com.metawebthree.warehouse.infrastructure.persistence.repository.WarehouseRepository;
import com.metawebthree.warehouse.infrastructure.persistence.repository.InboundOrderRepository;
import org.springframework.stereotype.Service;
import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
public class WarehouseApplicationServiceImpl implements WarehouseApplicationService {

    // 状态常量
    private static final String STATUS_ACTIVE = "ACTIVE";
    private static final String STATUS_PENDING = "PENDING";
    
    // 编码前缀常量
    private static final String WAREHOUSE_CODE_PREFIX = "WH";
    private static final String INBOUND_ORDER_PREFIX = "IB";
    private static final int CODE_SUFFIX_LENGTH = 4;
    
    // 默认值常量
    private static final int DEFAULT_USED_CAPACITY = 0;
    private static final int DEFAULT_QUANTITY = 0;
    
    private final WarehouseRepository warehouseRepository;
    private final InboundOrderRepository inboundOrderRepository;
    private final WarehouseDomainEventPublisher eventPublisher;

    public WarehouseApplicationServiceImpl(WarehouseRepository warehouseRepository,
                                           InboundOrderRepository inboundOrderRepository,
                                           WarehouseDomainEventPublisher eventPublisher) {
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
        warehouse.setUsedCapacity(DEFAULT_USED_CAPACITY);
        warehouse.setStatus(STATUS_ACTIVE);
        warehouse.setCreatedAt(LocalDateTime.now());
        warehouse.setUpdatedAt(LocalDateTime.now());
        
        warehouseRepository.insert(warehouse);
        eventPublisher.publishCreated(warehouse.getId(), warehouse.getWarehouseCode(), warehouse.getWarehouseName());
        
        return toWarehouseDTO(warehouseRepository.findById(warehouse.getId()).orElse(warehouse));
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
                warehouseRepository.update(warehouse);
                return toWarehouseDTO(warehouseRepository.findById(id).orElse(warehouse));
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
        return List.of();
    }

    @Override
    public InboundOrderDTO createInboundOrder(InboundOrderDTO dto) {
        InboundOrder order = new InboundOrder();
        order.setOrderNo(generateInboundOrderNo());
        order.setInboundType(dto.getInboundType());
        order.setWarehouseId(dto.getWarehouseId());
        order.setSupplierCode(dto.getSupplierCode());
        order.setStatus(STATUS_PENDING);
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
                    item.setStatus(STATUS_PENDING);
                    return item;
                })
                .collect(Collectors.toList());
            order.setItems(items);
        }
        
        inboundOrderRepository.insert(order);
        eventPublisher.publishInboundOrderCreated(order.getOrderNo(), order.getWarehouseId());
        
        return toInboundOrderDTO(inboundOrderRepository.findById(order.getId()).orElse(order));
    }

    @Override
    public InboundOrderDTO confirmInboundOrder(String orderNo) {
        return inboundOrderRepository.findByOrderNo(orderNo)
            .map(order -> {
                order.confirm();
                order.setUpdatedAt(LocalDateTime.now());
                inboundOrderRepository.update(order);
                return toInboundOrderDTO(inboundOrderRepository.findByOrderNo(orderNo).orElse(order));
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
                inboundOrderRepository.update(order);
                
                eventPublisher.publishInboundOrderCompleted(orderNo);
                if (order.getItems() != null) {
                    for (InboundOrderItem item : order.getItems()) {
                        int qty = getActualQuantity(item);
                        eventPublisher.publishStockIn(
                            order.getWarehouseId(),
                            item.getSkuCode(),
                            qty,
                            orderNo
                        );
                    }
                }
                
                return toInboundOrderDTO(inboundOrderRepository.findByOrderNo(orderNo).orElse(order));
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
        return WAREHOUSE_CODE_PREFIX + System.currentTimeMillis() 
            + UUID.randomUUID().toString().substring(0, CODE_SUFFIX_LENGTH).toUpperCase();
    }

    private String generateInboundOrderNo() {
        return INBOUND_ORDER_PREFIX + System.currentTimeMillis() 
            + UUID.randomUUID().toString().substring(0, CODE_SUFFIX_LENGTH).toUpperCase();
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
        return dto;
    }

    private int getActualQuantity(InboundOrderItem item) {
        if (item.getActualQuantity() != null) {
            return item.getActualQuantity();
        }
        if (item.getPlanQuantity() != null) {
            return item.getPlanQuantity();
        }
        return DEFAULT_QUANTITY;
    }
}        dto.setExpiryDate(item.getExpiryDate());
        return dto;
    }
}