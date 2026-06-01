package com.metawebthree.inventoryalert.infrastructure.persistence.repository;

import com.metawebthree.inventoryalert.domain.model.InventoryAlertDO;
import com.metawebthree.inventoryalert.domain.repository.InventoryAlertRepository;
import com.metawebthree.inventoryalert.infrastructure.persistence.mapper.InventoryAlertMapper;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public class InventoryAlertRepositoryImpl implements InventoryAlertRepository {

    private final InventoryAlertMapper alertMapper;

    public InventoryAlertRepositoryImpl(InventoryAlertMapper alertMapper) {
        this.alertMapper = alertMapper;
    }

    @Override
    public InventoryAlertDO save(InventoryAlertDO alert) {
        if (alert.getId() == null) {
            alertMapper.insert(alert);
        } else {
            alertMapper.update(alert);
        }
        return alert;
    }

    @Override
    public InventoryAlertDO findById(Long id) {
        return alertMapper.selectById(id);
    }

    @Override
    public List<InventoryAlertDO> findByProductId(Long productId) {
        return alertMapper.selectByProductId(productId);
    }

    @Override
    public List<InventoryAlertDO> findByWarehouseId(Long warehouseId) {
        return alertMapper.selectByWarehouseId(warehouseId);
    }

    @Override
    public List<InventoryAlertDO> findByAlertLevel(Integer alertLevel) {
        return alertMapper.selectByAlertLevel(alertLevel);
    }

    @Override
    public List<InventoryAlertDO> findByStatus(Integer status) {
        return alertMapper.selectByStatus(status);
    }

    @Override
    public List<InventoryAlertDO> findAll() {
        return alertMapper.selectAll();
    }

    @Override
    public boolean updateStatus(Long id, Integer status) {
        return alertMapper.updateStatus(id, status) > 0;
    }

    @Override
    public boolean deleteById(Long id) {
        return alertMapper.deleteById(id) > 0;
    }
}