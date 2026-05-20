package com.metawebthree.procurement.infrastructure.persistence.repository;

import com.metawebthree.procurement.domain.entity.ProcurementOrder;
import com.metawebthree.procurement.domain.repository.ProcurementOrderRepository;
import org.springframework.stereotype.Repository;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

@Repository
public class ProcurementOrderRepositoryImpl implements ProcurementOrderRepository {

    private final Map<Long, ProcurementOrder> store = new ConcurrentHashMap<>();
    private final Map<String, ProcurementOrder> orderNoIndex = new ConcurrentHashMap<>();
    private Long idSequence = 1L;

    @Override
    public ProcurementOrder save(ProcurementOrder order) {
        if (order.getId() == null) {
            order.setId(idSequence++);
            order.setUpdatedAt(java.time.LocalDateTime.now());
        }
        store.put(order.getId(), order);
        orderNoIndex.put(order.getOrderNo(), order);
        return order;
    }

    @Override
    public Optional<ProcurementOrder> findById(Long id) {
        return Optional.ofNullable(store.get(id));
    }

    @Override
    public Optional<ProcurementOrder> findByOrderNo(String orderNo) {
        return Optional.ofNullable(orderNoIndex.get(orderNo));
    }

    @Override
    public List<ProcurementOrder> findByStatus(String status) {
        return store.values().stream()
            .filter(o -> status.equals(o.getStatus()))
            .toList();
    }

    @Override
    public List<ProcurementOrder> findBySupplierCode(String supplierCode) {
        return store.values().stream()
            .filter(o -> supplierCode.equals(o.getSupplierCode()))
            .toList();
    }

    @Override
    public List<ProcurementOrder> findByWarehouseId(Long warehouseId) {
        return store.values().stream()
            .filter(o -> warehouseId.equals(o.getWarehouseId()))
            .toList();
    }
}