package com.metawebthree.digitaltwin.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.digitaltwin.domain.entity.InventoryAlert;
import com.metawebthree.digitaltwin.domain.entity.InventoryAlert.AlertLevel;
import com.metawebthree.digitaltwin.domain.entity.InventoryAlert.AlertStatus;
import com.metawebthree.digitaltwin.domain.repository.InventoryAlertRepository;
import com.metawebthree.digitaltwin.infrastructure.persistence.converter.InventoryAlertConverter;
import com.metawebthree.digitaltwin.infrastructure.persistence.dataobject.InventoryAlertDO;
import com.metawebthree.digitaltwin.infrastructure.persistence.mapper.InventoryAlertMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class InventoryAlertRepositoryImpl implements InventoryAlertRepository {

    private final InventoryAlertMapper inventoryAlertMapper;
    private final InventoryAlertConverter inventoryAlertConverter;

    public InventoryAlertRepositoryImpl(InventoryAlertMapper inventoryAlertMapper,
                                         InventoryAlertConverter inventoryAlertConverter) {
        this.inventoryAlertMapper = inventoryAlertMapper;
        this.inventoryAlertConverter = inventoryAlertConverter;
    }

    @Override
    public Optional<InventoryAlert> findById(Long id) {
        InventoryAlertDO alertDO = inventoryAlertMapper.selectById(id);
        return Optional.ofNullable(inventoryAlertConverter.toEntity(alertDO));
    }

    @Override
    public Optional<InventoryAlert> findByAlertCode(String alertCode) {
        LambdaQueryWrapper<InventoryAlertDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InventoryAlertDO::getAlertCode, alertCode);
        InventoryAlertDO alertDO = inventoryAlertMapper.selectOne(wrapper);
        return Optional.ofNullable(inventoryAlertConverter.toEntity(alertDO));
    }

    @Override
    public List<InventoryAlert> findAll() {
        return inventoryAlertMapper.selectList(null).stream()
                .map(inventoryAlertConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<InventoryAlert> findByWarehouseCode(String warehouseCode) {
        LambdaQueryWrapper<InventoryAlertDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InventoryAlertDO::getWarehouseCode, warehouseCode);
        return inventoryAlertMapper.selectList(wrapper).stream()
                .map(inventoryAlertConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<InventoryAlert> findByItemCode(String itemCode) {
        LambdaQueryWrapper<InventoryAlertDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InventoryAlertDO::getItemCode, itemCode);
        return inventoryAlertMapper.selectList(wrapper).stream()
                .map(inventoryAlertConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<InventoryAlert> findByStatus(AlertStatus status) {
        LambdaQueryWrapper<InventoryAlertDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InventoryAlertDO::getStatus, status.name());
        return inventoryAlertMapper.selectList(wrapper).stream()
                .map(inventoryAlertConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<InventoryAlert> findByLevel(AlertLevel level) {
        LambdaQueryWrapper<InventoryAlertDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InventoryAlertDO::getLevel, level.name());
        return inventoryAlertMapper.selectList(wrapper).stream()
                .map(inventoryAlertConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<InventoryAlert> findActiveAlerts() {
        LambdaQueryWrapper<InventoryAlertDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.in(InventoryAlertDO::getStatus,
                   AlertStatus.TRIGGERED.name(),
                   AlertStatus.ACKNOWLEDGED.name(),
                   AlertStatus.IN_PROGRESS.name());
        return inventoryAlertMapper.selectList(wrapper).stream()
                .map(inventoryAlertConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<InventoryAlert> findByWarehouseCodeAndStatus(String warehouseCode, AlertStatus status) {
        LambdaQueryWrapper<InventoryAlertDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InventoryAlertDO::getWarehouseCode, warehouseCode)
               .eq(InventoryAlertDO::getStatus, status.name());
        return inventoryAlertMapper.selectList(wrapper).stream()
                .map(inventoryAlertConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public InventoryAlert save(InventoryAlert inventoryAlert) {
        InventoryAlertDO alertDO = inventoryAlertConverter.toDO(inventoryAlert);
        if (inventoryAlert.getId() == null) {
            inventoryAlertMapper.insert(alertDO);
            inventoryAlert.setId(alertDO.getId());
        } else {
            inventoryAlertMapper.updateById(alertDO);
        }
        return inventoryAlert;
    }

    @Override
    public void delete(InventoryAlert inventoryAlert) {
        if (inventoryAlert.getId() != null) {
            inventoryAlertMapper.deleteById(inventoryAlert.getId());
        }
    }

    @Override
    public boolean existsByAlertCode(String alertCode) {
        LambdaQueryWrapper<InventoryAlertDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InventoryAlertDO::getAlertCode, alertCode);
        return inventoryAlertMapper.selectCount(wrapper) > 0;
    }
}