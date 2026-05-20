package com.metawebthree.procurement.domain.service;

import com.metawebthree.procurement.domain.entity.ProcurementOrder;
import com.metawebthree.procurement.domain.repository.ProcurementOrderRepository;
import org.springframework.stereotype.Service;
import java.util.Optional;

@Service
public class ProcurementDomainService {

    private final ProcurementOrderRepository repository;

    public ProcurementDomainService(ProcurementOrderRepository repository) {
        this.repository = repository;
    }

    public ProcurementOrder createOrder(String supplierCode, Long warehouseId, String purchaseType) {
        ProcurementOrder order = new ProcurementOrder();
        order.setOrderNo(generateOrderNo());
        order.setSupplierCode(supplierCode);
        order.setWarehouseId(warehouseId);
        order.setPurchaseType(purchaseType);
        order.setStatus("PENDING");
        order.setCreatedAt(java.time.LocalDateTime.now());
        return repository.save(order);
    }

    public Optional<ProcurementOrder> findByOrderNo(String orderNo) {
        return repository.findByOrderNo(orderNo);
    }

    public ProcurementOrder approve(ProcurementOrder order, String approver) {
        order.approve(approver);
        return repository.save(order);
    }

    public ProcurementOrder reject(ProcurementOrder order, String reason) {
        order.reject(reason);
        return repository.save(order);
    }

    public ProcurementOrder complete(ProcurementOrder order) {
        order.complete();
        return repository.save(order);
    }

    public ProcurementOrder cancel(ProcurementOrder order) {
        order.cancel();
        return repository.save(order);
    }

    private String generateOrderNo() {
        return "PO" + System.currentTimeMillis();
    }
}