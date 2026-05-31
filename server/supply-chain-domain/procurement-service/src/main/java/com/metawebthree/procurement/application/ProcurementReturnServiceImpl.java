package com.metawebthree.procurement.application;

import com.metawebthree.procurement.application.dto.ProcurementReturnOrderDTO;
import com.metawebthree.procurement.application.dto.ProcurementReturnOrderItemDTO;
import com.metawebthree.procurement.domain.entity.ProcurementReturnOrder;
import com.metawebthree.procurement.domain.entity.ProcurementReturnOrderItem;
import com.metawebthree.procurement.domain.repository.ProcurementReturnOrderRepository;
import com.metawebthree.procurement.infrastructure.event.ProcurementReturnDomainEventPublisher;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
public class ProcurementReturnServiceImpl implements ProcurementReturnService {
    
    private static final String RETURN_ORDER_PREFIX = "RPO";
    
    private final ProcurementReturnOrderRepository returnOrderRepository;
    private final ProcurementReturnDomainEventPublisher eventPublisher;
    
    public ProcurementReturnServiceImpl(
            ProcurementReturnOrderRepository returnOrderRepository,
            ProcurementReturnDomainEventPublisher eventPublisher) {
        this.returnOrderRepository = returnOrderRepository;
        this.eventPublisher = eventPublisher;
    }
    
    @Override
    @Transactional
    public ProcurementReturnOrderDTO createReturnOrder(ProcurementReturnOrderDTO dto) {
        ProcurementReturnOrder order = new ProcurementReturnOrder();
        order.setReturnNo(generateReturnNo());
        order.setSourceOrderNo(dto.getSourceOrderNo());
        order.setSourceOrderType(dto.getSourceOrderType() != null ? dto.getSourceOrderType() : "PROCUREMENT");
        order.setSupplierCode(dto.getSupplierCode());
        order.setSupplierName(dto.getSupplierName());
        order.setWarehouseId(dto.getWarehouseId());
        order.setWarehouseName(dto.getWarehouseName());
        order.setReturnType(dto.getReturnType());
        order.setStatus(ProcurementReturnOrder.STATUS_DRAFT);
        order.setTotalAmount(dto.getTotalAmount());
        order.setCurrency(dto.getCurrency() != null ? dto.getCurrency() : "CNY");
        order.setReason(dto.getReason());
        order.setRemark(dto.getRemark());
        order.setExpectedReturnDate(dto.getExpectedReturnDate());
        order.setCreatedAt(LocalDateTime.now());
        order.setUpdatedAt(LocalDateTime.now());
        
        if (dto.getItems() != null && !dto.getItems().isEmpty()) {
            List<ProcurementReturnOrderItem> items = dto.getItems().stream()
                .map(this::toItemEntity)
                .collect(Collectors.toList());
            order.setItems(items);
        }
        
        returnOrderRepository.insert(order);
        eventPublisher.publishReturnOrderCreated(order.getReturnNo(), order.getSourceOrderNo());
        
        return toDTO(order);
    }
    
    @Override
    @Transactional
    public ProcurementReturnOrderDTO submitForApproval(String returnNo) {
        ProcurementReturnOrder order = returnOrderRepository.findByReturnNo(returnNo)
            .orElseThrow(() -> new RuntimeException("Return order not found: " + returnNo));
        
        order.submitForApproval();
        order.setUpdatedAt(LocalDateTime.now());
        returnOrderRepository.update(order);
        
        eventPublisher.publishReturnOrderSubmitted(returnNo);
        return toDTO(order);
    }
    
    @Override
    @Transactional
    public ProcurementReturnOrderDTO approveReturnOrder(String returnNo, String approver, String comment) {
        ProcurementReturnOrder order = returnOrderRepository.findByReturnNo(returnNo)
            .orElseThrow(() -> new RuntimeException("Return order not found: " + returnNo));
        
        order.approve(approver, comment);
        order.setUpdatedAt(LocalDateTime.now());
        returnOrderRepository.update(order);
        
        eventPublisher.publishReturnOrderApproved(returnNo);
        return toDTO(order);
    }
    
    @Override
    @Transactional
    public ProcurementReturnOrderDTO rejectReturnOrder(String returnNo, String approver, String reason) {
        ProcurementReturnOrder order = returnOrderRepository.findByReturnNo(returnNo)
            .orElseThrow(() -> new RuntimeException("Return order not found: " + returnNo));
        
        order.reject(approver, reason);
        order.setUpdatedAt(LocalDateTime.now());
        returnOrderRepository.update(order);
        
        eventPublisher.publishReturnOrderRejected(returnNo);
        return toDTO(order);
    }
    
    @Override
    @Transactional
    public ProcurementReturnOrderDTO shipReturnOrder(String returnNo, String logisticsCompany, String trackingNumber) {
        ProcurementReturnOrder order = returnOrderRepository.findByReturnNo(returnNo)
            .orElseThrow(() -> new RuntimeException("Return order not found: " + returnNo));
        
        order.ship(logisticsCompany, trackingNumber);
        order.setUpdatedAt(LocalDateTime.now());
        returnOrderRepository.update(order);
        
        eventPublisher.publishReturnOrderShipped(returnNo, logisticsCompany, trackingNumber);
        return toDTO(order);
    }
    
    @Override
    @Transactional
    public ProcurementReturnOrderDTO confirmReturned(String returnNo) {
        ProcurementReturnOrder order = returnOrderRepository.findByReturnNo(returnNo)
            .orElseThrow(() -> new RuntimeException("Return order not found: " + returnNo));
        
        order.confirmReturned();
        order.setUpdatedAt(LocalDateTime.now());
        returnOrderRepository.update(order);
        
        eventPublisher.publishReturnOrderConfirmed(returnNo);
        return toDTO(order);
    }
    
    @Override
    @Transactional
    public ProcurementReturnOrderDTO completeReturnOrder(String returnNo) {
        ProcurementReturnOrder order = returnOrderRepository.findByReturnNo(returnNo)
            .orElseThrow(() -> new RuntimeException("Return order not found: " + returnNo));
        
        order.complete();
        order.setUpdatedAt(LocalDateTime.now());
        returnOrderRepository.update(order);
        
        eventPublisher.publishReturnOrderCompleted(returnNo);
        return toDTO(order);
    }
    
    @Override
    @Transactional
    public ProcurementReturnOrderDTO cancelReturnOrder(String returnNo) {
        ProcurementReturnOrder order = returnOrderRepository.findByReturnNo(returnNo)
            .orElseThrow(() -> new RuntimeException("Return order not found: " + returnNo));
        
        order.cancel();
        order.setUpdatedAt(LocalDateTime.now());
        returnOrderRepository.update(order);
        
        eventPublisher.publishReturnOrderCancelled(returnNo);
        return toDTO(order);
    }
    
    @Override
    public ProcurementReturnOrderDTO queryReturnOrder(String returnNo) {
        return returnOrderRepository.findByReturnNo(returnNo)
            .map(this::toDTO)
            .orElseThrow(() -> new RuntimeException("Return order not found: " + returnNo));
    }
    
    @Override
    public List<ProcurementReturnOrderDTO> listReturnOrders(String status, Long warehouseId, String supplierCode) {
        List<ProcurementReturnOrder> orders;
        
        if (status != null && !status.isEmpty()) {
            orders = returnOrderRepository.findByStatus(status);
        } else if (warehouseId != null) {
            orders = returnOrderRepository.findByWarehouseId(warehouseId);
        } else if (supplierCode != null && !supplierCode.isEmpty()) {
            orders = returnOrderRepository.findBySupplierCode(supplierCode);
        } else {
            orders = returnOrderRepository.findAll();
        }
        
        return orders.stream()
            .map(this::toDTO)
            .collect(Collectors.toList());
    }
    
    private String generateReturnNo() {
        return RETURN_ORDER_PREFIX + System.currentTimeMillis() 
            + UUID.randomUUID().toString().substring(0, 4).toUpperCase();
    }
    
    private ProcurementReturnOrderDTO toDTO(ProcurementReturnOrder order) {
        ProcurementReturnOrderDTO dto = new ProcurementReturnOrderDTO();
        dto.setId(order.getId());
        dto.setReturnNo(order.getReturnNo());
        dto.setSourceOrderNo(order.getSourceOrderNo());
        dto.setSourceOrderType(order.getSourceOrderType());
        dto.setSupplierCode(order.getSupplierCode());
        dto.setSupplierName(order.getSupplierName());
        dto.setWarehouseId(order.getWarehouseId());
        dto.setWarehouseName(order.getWarehouseName());
        dto.setReturnType(order.getReturnType());
        dto.setStatus(order.getStatus());
        dto.setTotalAmount(order.getTotalAmount());
        dto.setCurrency(order.getCurrency());
        dto.setReason(order.getReason());
        dto.setRemark(order.getRemark());
        dto.setApprover(order.getApprover());
        dto.setApprovalComment(order.getApprovalComment());
        dto.setApprovedAt(order.getApprovedAt());
        dto.setExpectedReturnDate(order.getExpectedReturnDate());
        dto.setActualReturnDate(order.getActualReturnDate());
        dto.setLogisticsCompany(order.getLogisticsCompany());
        dto.setTrackingNumber(order.getTrackingNumber());
        dto.setShippedAt(order.getShippedAt());
        dto.setCreatedAt(order.getCreatedAt());
        dto.setUpdatedAt(order.getUpdatedAt());
        
        if (order.getItems() != null) {
            dto.setItems(order.getItems().stream()
                .map(this::toItemDTO)
                .collect(Collectors.toList()));
        }
        return dto;
    }
    
    private ProcurementReturnOrderItemDTO toItemDTO(ProcurementReturnOrderItem item) {
        ProcurementReturnOrderItemDTO dto = new ProcurementReturnOrderItemDTO();
        dto.setId(item.getId());
        dto.setReturnOrderId(item.getReturnOrderId());
        dto.setReturnNo(item.getReturnNo());
        dto.setSourceOrderNo(item.getSourceOrderNo());
        dto.setSourceOrderItemId(item.getSourceOrderItemId());
        dto.setSkuCode(item.getSkuCode());
        dto.setProductName(item.getProductName());
        dto.setReturnQuantity(item.getReturnQuantity());
        dto.setUnitPrice(item.getUnitPrice());
        dto.setTotalAmount(item.getTotalAmount());
        dto.setReason(item.getReason());
        dto.setStatus(item.getStatus());
        return dto;
    }
    
    private ProcurementReturnOrderItem toItemEntity(ProcurementReturnOrderItemDTO dto) {
        ProcurementReturnOrderItem item = new ProcurementReturnOrderItem();
        item.setId(dto.getId());
        item.setReturnOrderId(dto.getReturnOrderId());
        item.setReturnNo(dto.getReturnNo());
        item.setSourceOrderNo(dto.getSourceOrderNo());
        item.setSourceOrderItemId(dto.getSourceOrderItemId());
        item.setSkuCode(dto.getSkuCode());
        item.setProductName(dto.getProductName());
        item.setReturnQuantity(dto.getReturnQuantity());
        item.setUnitPrice(dto.getUnitPrice());
        item.setTotalAmount(dto.getTotalAmount());
        item.setReason(dto.getReason());
        item.setStatus(ProcurementReturnOrderItem.STATUS_PENDING);
        return item;
    }
}