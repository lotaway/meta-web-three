package com.metawebthree.procurement.application;

import com.metawebthree.procurement.application.dto.ProcurementOrderDTO;
import com.metawebthree.procurement.domain.entity.ProcurementOrder;
import com.metawebthree.procurement.domain.repository.ProcurementOrderRepository;
import com.metawebthree.procurement.infrastructure.event.ProcurementDomainEventPublisher;
import org.springframework.stereotype.Service;
import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
public class ProcurementApplicationServiceImpl implements ProcurementApplicationService {

    private final ProcurementOrderRepository repository;
    private final ProcurementDomainEventPublisher eventPublisher;

    public ProcurementApplicationServiceImpl(ProcurementOrderRepository repository,
                                              ProcurementDomainEventPublisher eventPublisher) {
        this.repository = repository;
        this.eventPublisher = eventPublisher;
    }

    @Override
    public ProcurementOrderDTO createOrder(ProcurementOrderDTO dto) {
        ProcurementOrder order = new ProcurementOrder();
        order.setOrderNo(generateOrderNo());
        order.setSupplierCode(dto.getSupplierCode());
        order.setWarehouseId(dto.getWarehouseId());
        order.setPurchaseType(dto.getPurchaseType());
        order.setStatus("PENDING");
        order.setTotalAmount(dto.getTotalAmount());
        order.setCurrency(dto.getCurrency() != null ? dto.getCurrency() : "CNY");
        order.setPaymentTerms(dto.getPaymentTerms());
        order.setDeliveryTerms(dto.getDeliveryTerms());
        order.setRemark(dto.getRemark());
        order.setExpectedDeliveryDate(dto.getExpectedDeliveryDate());
        order.setCreatedAt(LocalDateTime.now());
        order.setUpdatedAt(LocalDateTime.now());
        
        ProcurementOrder saved = repository.save(order);
        eventPublisher.publishCreated(saved.getOrderNo(), saved.getSupplierCode());
        
        return toDTO(saved);
    }

    @Override
    public ProcurementOrderDTO approveOrder(String orderNo, String approver) {
        return repository.findByOrderNo(orderNo)
            .map(order -> {
                order.approve(approver);
                order.setUpdatedAt(LocalDateTime.now());
                ProcurementOrder updated = repository.save(order);
                eventPublisher.publishApproved(orderNo, approver);
                return toDTO(updated);
            })
            .orElse(null);
    }

    @Override
    public ProcurementOrderDTO rejectOrder(String orderNo, String reason) {
        return repository.findByOrderNo(orderNo)
            .map(order -> {
                order.reject(reason);
                order.setUpdatedAt(LocalDateTime.now());
                ProcurementOrder updated = repository.save(order);
                eventPublisher.publishRejected(orderNo, reason);
                return toDTO(updated);
            })
            .orElse(null);
    }

    @Override
    public ProcurementOrderDTO queryOrder(String orderNo) {
        return repository.findByOrderNo(orderNo)
            .map(this::toDTO)
            .orElse(null);
    }

    @Override
    public List<ProcurementOrderDTO> listOrders(String status) {
        List<ProcurementOrder> orders;
        if (status != null && !status.isEmpty()) {
            orders = repository.findByStatus(status);
        } else {
            orders = List.of();
        }
        return orders.stream()
            .map(this::toDTO)
            .collect(Collectors.toList());
    }

    public ProcurementOrderDTO cancelOrder(String orderNo) {
        return repository.findByOrderNo(orderNo)
            .map(order -> {
                order.cancel();
                order.setUpdatedAt(LocalDateTime.now());
                ProcurementOrder updated = repository.save(order);
                eventPublisher.publishCancelled(orderNo);
                return toDTO(updated);
            })
            .orElse(null);
    }

    public ProcurementOrderDTO completeOrder(String orderNo) {
        return repository.findByOrderNo(orderNo)
            .map(order -> {
                order.complete();
                order.setUpdatedAt(LocalDateTime.now());
                ProcurementOrder updated = repository.save(order);
                eventPublisher.publishCompleted(orderNo);
                return toDTO(updated);
            })
            .orElse(null);
    }

    public List<ProcurementOrderDTO> listBySupplier(String supplierCode) {
        return repository.findBySupplierCode(supplierCode).stream()
            .map(this::toDTO)
            .collect(Collectors.toList());
    }

    public List<ProcurementOrderDTO> listByWarehouse(Long warehouseId) {
        return repository.findByWarehouseId(warehouseId).stream()
            .map(this::toDTO)
            .collect(Collectors.toList());
    }

    private String generateOrderNo() {
        return "PO" + System.currentTimeMillis() + UUID.randomUUID().toString().substring(0, 6).toUpperCase();
    }

    private ProcurementOrderDTO toDTO(ProcurementOrder order) {
        ProcurementOrderDTO dto = new ProcurementOrderDTO();
        dto.setId(order.getId());
        dto.setOrderNo(order.getOrderNo());
        dto.setSupplierCode(order.getSupplierCode());
        dto.setWarehouseId(order.getWarehouseId());
        dto.setPurchaseType(order.getPurchaseType());
        dto.setStatus(order.getStatus());
        dto.setTotalAmount(order.getTotalAmount());
        dto.setCurrency(order.getCurrency());
        dto.setPaymentTerms(order.getPaymentTerms());
        dto.setDeliveryTerms(order.getDeliveryTerms());
        dto.setRemark(order.getRemark());
        dto.setApprover(order.getApprover());
        dto.setApprovedAt(order.getApprovedAt());
        dto.setExpectedDeliveryDate(order.getExpectedDeliveryDate());
        dto.setActualDeliveryDate(order.getActualDeliveryDate());
        dto.setCreatedAt(order.getCreatedAt());
        return dto;
    }
}