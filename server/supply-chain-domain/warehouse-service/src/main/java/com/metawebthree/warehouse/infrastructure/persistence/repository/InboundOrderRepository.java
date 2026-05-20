package com.metawebthree.warehouse.infrastructure.persistence.repository;

import com.metawebthree.warehouse.domain.entity.InboundOrder;
import java.util.Optional;
import java.util.List;

public interface InboundOrderRepository {

    Optional<InboundOrder> findById(Long id);

    Optional<InboundOrder> findByOrderNo(String orderNo);

    InboundOrder save(InboundOrder order);

    List<InboundOrder> findByWarehouseId(Long warehouseId);

    List<InboundOrder> findByStatus(String status);
}