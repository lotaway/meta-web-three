package com.metawebthree.supplier.infrastructure.persistence.repository;

import com.metawebthree.supplier.domain.entity.SupplierReconciliation;
import com.metawebthree.supplier.domain.entity.SupplierReconciliationItem;
import com.metawebthree.supplier.domain.repository.SupplierReconciliationRepository;
import com.metawebthree.supplier.infrastructure.persistence.dataobject.SupplierReconciliationDO;
import com.metawebthree.supplier.infrastructure.persistence.dataobject.SupplierReconciliationItemDO;
import com.metawebthree.supplier.infrastructure.persistence.mapper.SupplierReconciliationItemMapper;
import com.metawebthree.supplier.infrastructure.persistence.mapper.SupplierReconciliationMapper;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public class SupplierReconciliationRepositoryImpl implements SupplierReconciliationRepository {

    private final SupplierReconciliationMapper reconciliationMapper;
    private final SupplierReconciliationItemMapper reconciliationItemMapper;

    public SupplierReconciliationRepositoryImpl(
            SupplierReconciliationMapper reconciliationMapper,
            SupplierReconciliationItemMapper reconciliationItemMapper) {
        this.reconciliationMapper = reconciliationMapper;
        this.reconciliationItemMapper = reconciliationItemMapper;
    }

    @Override
    public SupplierReconciliation save(SupplierReconciliation reconciliation) {
        SupplierReconciliationDO reconDO = toDO(reconciliation);
        if (reconciliation.getId() == null) {
            reconDO.setCreatedAt(LocalDateTime.now());
            reconciliationMapper.insert(reconDO);
            reconciliation.setId(reconDO.getId());
        } else {
            reconDO.setUpdatedAt(LocalDateTime.now());
            reconciliationMapper.updateById(reconDO);
        }
        
        // 保存明细
        if (reconciliation.getItems() != null && !reconciliation.getItems().isEmpty()) {
            for (SupplierReconciliationItem item : reconciliation.getItems()) {
                SupplierReconciliationItemDO itemDO = toItemDO(item);
                itemDO.setReconciliationId(reconciliation.getId());
                if (item.getId() == null) {
                    itemDO.setCreatedAt(LocalDateTime.now());
                    reconciliationItemMapper.insert(itemDO);
                    item.setId(itemDO.getId());
                } else {
                    itemDO.setUpdatedAt(LocalDateTime.now());
                    reconciliationItemMapper.updateById(itemDO);
                }
            }
        }
        
        return reconciliation;
    }

    @Override
    public Optional<SupplierReconciliation> findById(Long id) {
        SupplierReconciliationDO reconDO = reconciliationMapper.selectById(id);
        if (reconDO == null) {
            return Optional.empty();
        }
        SupplierReconciliation reconciliation = toEntity(reconDO);
        List<SupplierReconciliationItemDO> itemDOs = reconciliationItemMapper.selectByReconciliationId(id);
        reconciliation.setItems(itemDOs.stream().map(this::toItemEntity).toList());
        return Optional.of(reconciliation);
    }

    @Override
    public Optional<SupplierReconciliation> findByReconciliationNo(String reconciliationNo) {
        SupplierReconciliationDO reconDO = reconciliationMapper.selectByReconciliationNo(reconciliationNo);
        if (reconDO == null) {
            return Optional.empty();
        }
        SupplierReconciliation reconciliation = toEntity(reconDO);
        List<SupplierReconciliationItemDO> itemDOs = reconciliationItemMapper.selectByReconciliationId(reconDO.getId());
        reconciliation.setItems(itemDOs.stream().map(this::toItemEntity).toList());
        return Optional.of(reconciliation);
    }

    @Override
    public List<SupplierReconciliation> findBySupplierCode(String supplierCode) {
        List<SupplierReconciliationDO> list = reconciliationMapper.selectBySupplierCode(supplierCode);
        return list.stream().map(this::toEntity).toList();
    }

    @Override
    public List<SupplierReconciliation> findBySupplierCodeAndStatus(String supplierCode, String status) {
        List<SupplierReconciliationDO> list = reconciliationMapper.selectBySupplierCodeAndStatus(supplierCode, status);
        return list.stream().map(this::toEntity).toList();
    }

    @Override
    public List<SupplierReconciliation> findByPeriod(String supplierCode, LocalDate periodStart, LocalDate periodEnd) {
        List<SupplierReconciliationDO> list = reconciliationMapper.selectByPeriod(supplierCode, periodStart, periodEnd);
        return list.stream().map(this::toEntity).toList();
    }

    @Override
    public List<SupplierReconciliation> findByStatus(String status) {
        List<SupplierReconciliationDO> list = reconciliationMapper.selectByStatus(status);
        return list.stream().map(this::toEntity).toList();
    }

    @Override
    public void deleteById(Long id) {
        reconciliationItemMapper.deleteById(id);
        reconciliationMapper.deleteById(id);
    }

    private SupplierReconciliationDO toDO(SupplierReconciliation reconciliation) {
        SupplierReconciliationDO doItem = new SupplierReconciliationDO();
        doItem.setId(reconciliation.getId());
        doItem.setReconciliationNo(reconciliation.getReconciliationNo());
        doItem.setSupplierCode(reconciliation.getSupplierCode());
        doItem.setPeriodStart(reconciliation.getPeriodStart());
        doItem.setPeriodEnd(reconciliation.getPeriodEnd());
        doItem.setOrderCount(reconciliation.getOrderCount());
        doItem.setTotalAmount(reconciliation.getTotalAmount());
        doItem.setShippedAmount(reconciliation.getShippedAmount());
        doItem.setInvoicedAmount(reconciliation.getInvoicedAmount());
        doItem.setSettledAmount(reconciliation.getSettledAmount());
        doItem.setPendingAmount(reconciliation.getPendingAmount());
        doItem.setCurrency(reconciliation.getCurrency());
        doItem.setStatus(reconciliation.getStatus());
        doItem.setSubmittedAt(reconciliation.getSubmittedAt());
        doItem.setConfirmedAt(reconciliation.getConfirmedAt());
        doItem.setConfirmedBy(reconciliation.getConfirmedBy());
        doItem.setPaidAt(reconciliation.getPaidAt());
        doItem.setRemark(reconciliation.getRemark());
        doItem.setCreatedAt(reconciliation.getCreatedAt());
        doItem.setUpdatedAt(reconciliation.getUpdatedAt());
        return doItem;
    }

    private SupplierReconciliation toEntity(SupplierReconciliationDO doItem) {
        if (doItem == null) {
            return null;
        }
        SupplierReconciliation reconciliation = new SupplierReconciliation();
        reconciliation.setId(doItem.getId());
        reconciliation.setReconciliationNo(doItem.getReconciliationNo());
        reconciliation.setSupplierCode(doItem.getSupplierCode());
        reconciliation.setPeriodStart(doItem.getPeriodStart());
        reconciliation.setPeriodEnd(doItem.getPeriodEnd());
        reconciliation.setOrderCount(doItem.getOrderCount());
        reconciliation.setTotalAmount(doItem.getTotalAmount());
        reconciliation.setShippedAmount(doItem.getShippedAmount());
        reconciliation.setInvoicedAmount(doItem.getInvoicedAmount());
        reconciliation.setSettledAmount(doItem.getSettledAmount());
        reconciliation.setPendingAmount(doItem.getPendingAmount());
        reconciliation.setCurrency(doItem.getCurrency());
        reconciliation.setStatus(doItem.getStatus());
        reconciliation.setSubmittedAt(doItem.getSubmittedAt());
        reconciliation.setConfirmedAt(doItem.getConfirmedAt());
        reconciliation.setConfirmedBy(doItem.getConfirmedBy());
        reconciliation.setPaidAt(doItem.getPaidAt());
        reconciliation.setRemark(doItem.getRemark());
        reconciliation.setCreatedAt(doItem.getCreatedAt());
        reconciliation.setUpdatedAt(doItem.getUpdatedAt());
        return reconciliation;
    }

    private SupplierReconciliationItemDO toItemDO(SupplierReconciliationItem item) {
        SupplierReconciliationItemDO doItem = new SupplierReconciliationItemDO();
        doItem.setId(item.getId());
        doItem.setReconciliationId(item.getReconciliationId());
        doItem.setOrderNo(item.getOrderNo());
        doItem.setOrderDate(item.getOrderDate());
        doItem.setShippedDate(item.getShippedDate());
        doItem.setInvoicedAmount(item.getInvoicedAmount());
        doItem.setSettledAmount(item.getSettledAmount());
        doItem.setPendingAmount(item.getPendingAmount());
        doItem.setStatus(item.getStatus());
        doItem.setRemark(item.getRemark());
        doItem.setCreatedAt(item.getCreatedAt());
        doItem.setUpdatedAt(item.getUpdatedAt());
        return doItem;
    }

    private SupplierReconciliationItem toItemEntity(SupplierReconciliationItemDO doItem) {
        if (doItem == null) {
            return null;
        }
        SupplierReconciliationItem item = new SupplierReconciliationItem();
        item.setId(doItem.getId());
        item.setReconciliationId(doItem.getReconciliationId());
        item.setOrderNo(doItem.getOrderNo());
        item.setOrderDate(doItem.getOrderDate());
        item.setShippedDate(doItem.getShippedDate());
        item.setInvoicedAmount(doItem.getInvoicedAmount());
        item.setSettledAmount(doItem.getSettledAmount());
        item.setPendingAmount(doItem.getPendingAmount());
        item.setStatus(doItem.getStatus());
        item.setRemark(doItem.getRemark());
        item.setCreatedAt(doItem.getCreatedAt());
        item.setUpdatedAt(doItem.getUpdatedAt());
        return item;
    }
}