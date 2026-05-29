package com.metawebthree.procurement.infrastructure.persistence.repository;

import com.metawebthree.procurement.domain.entity.ProcurementOrder;
import com.metawebthree.procurement.domain.repository.ProcurementOrderRepository;
import com.metawebthree.procurement.infrastructure.persistence.dataobject.ProcurementOrderDO;
import com.metawebthree.procurement.infrastructure.persistence.mapper.ProcurementOrderMapper;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public class ProcurementOrderRepositoryImpl implements ProcurementOrderRepository {

    private final ProcurementOrderMapper procurementOrderMapper;

    public ProcurementOrderRepositoryImpl(ProcurementOrderMapper procurementOrderMapper) {
        this.procurementOrderMapper = procurementOrderMapper;
    }

    @Override
    public ProcurementOrder save(ProcurementOrder order) {
        ProcurementOrderDO orderDO = toDO(order);
        if (order.getId() == null) {
            orderDO.setCreatedAt(LocalDateTime.now());
            procurementOrderMapper.insert(orderDO);
            order.setId(orderDO.getId());
        } else {
            orderDO.setUpdatedAt(LocalDateTime.now());
            procurementOrderMapper.updateById(orderDO);
        }
        return order;
    }

    @Override
    public Optional<ProcurementOrder> findById(Long id) {
        ProcurementOrderDO orderDO = procurementOrderMapper.selectById(id);
        return Optional.ofNullable(toEntity(orderDO));
    }

    @Override
    public Optional<ProcurementOrder> findByOrderNo(String orderNo) {
        ProcurementOrderDO orderDO = procurementOrderMapper.selectByOrderNo(orderNo);
        return Optional.ofNullable(toEntity(orderDO));
    }

    @Override
    public List<ProcurementOrder> findByStatus(String status) {
        List<ProcurementOrderDO> list = procurementOrderMapper.selectByStatus(status);
        return list.stream().map(this::toEntity).toList();
    }

    @Override
    public List<ProcurementOrder> findBySupplierCode(String supplierCode) {
        List<ProcurementOrderDO> list = procurementOrderMapper.selectBySupplierCode(supplierCode);
        return list.stream().map(this::toEntity).toList();
    }

    @Override
    public List<ProcurementOrder> findByWarehouseId(Long warehouseId) {
        List<ProcurementOrderDO> list = procurementOrderMapper.selectByWarehouseId(warehouseId);
        return list.stream().map(this::toEntity).toList();
    }

    private ProcurementOrderDO toDO(ProcurementOrder order) {
        ProcurementOrderDO orderDO = new ProcurementOrderDO();
        orderDO.setId(order.getId());
        orderDO.setOrderNo(order.getOrderNo());
        orderDO.setSupplierCode(order.getSupplierCode());
        orderDO.setWarehouseId(order.getWarehouseId());
        orderDO.setPurchaseType(order.getPurchaseType());
        orderDO.setStatus(order.getStatus());
        orderDO.setTotalAmount(order.getTotalAmount());
        orderDO.setCurrency(order.getCurrency());
        orderDO.setPaymentTerms(order.getPaymentTerms());
        orderDO.setDeliveryTerms(order.getDeliveryTerms());
        orderDO.setRemark(order.getRemark());
        orderDO.setApprover(order.getApprover());
        orderDO.setApprovedAt(order.getApprovedAt());
        orderDO.setExpectedDeliveryDate(order.getExpectedDeliveryDate());
        orderDO.setActualDeliveryDate(order.getActualDeliveryDate());
        orderDO.setCreatedAt(order.getCreatedAt());
        orderDO.setUpdatedAt(order.getUpdatedAt());
        return orderDO;
    }

    private ProcurementOrder toEntity(ProcurementOrderDO orderDO) {
        if (orderDO == null) {
            return null;
        }
        ProcurementOrder order = new ProcurementOrder();
        order.setId(orderDO.getId());
        order.setOrderNo(orderDO.getOrderNo());
        order.setSupplierCode(orderDO.getSupplierCode());
        order.setWarehouseId(orderDO.getWarehouseId());
        order.setPurchaseType(orderDO.getPurchaseType());
        order.setStatus(orderDO.getStatus());
        order.setTotalAmount(orderDO.getTotalAmount());
        order.setCurrency(orderDO.getCurrency());
        order.setPaymentTerms(orderDO.getPaymentTerms());
        order.setDeliveryTerms(orderDO.getDeliveryTerms());
        order.setRemark(orderDO.getRemark());
        order.setApprover(orderDO.getApprover());
        order.setApprovedAt(orderDO.getApprovedAt());
        order.setExpectedDeliveryDate(orderDO.getExpectedDeliveryDate());
        order.setActualDeliveryDate(orderDO.getActualDeliveryDate());
        order.setCreatedAt(orderDO.getCreatedAt());
        order.setUpdatedAt(orderDO.getUpdatedAt());
        return order;
    }
}