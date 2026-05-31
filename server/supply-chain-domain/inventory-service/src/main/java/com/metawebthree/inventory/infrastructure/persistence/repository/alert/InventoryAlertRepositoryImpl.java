package com.metawebthree.inventory.infrastructure.persistence.repository.alert;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.inventory.domain.entity.alert.InventoryAlert;
import com.metawebthree.inventory.domain.repository.alert.InventoryAlertRepository;
import com.metawebthree.inventory.infrastructure.persistence.converter.InventoryAlertConverter;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.InventoryAlertDO;
import com.metawebthree.inventory.infrastructure.persistence.mapper.InventoryAlertMapper;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Repository
public class InventoryAlertRepositoryImpl implements InventoryAlertRepository {

    private final InventoryAlertMapper alertMapper;
    private final InventoryAlertConverter alertConverter;

    public InventoryAlertRepositoryImpl(InventoryAlertMapper alertMapper, InventoryAlertConverter alertConverter) {
        this.alertMapper = alertMapper;
        this.alertConverter = alertConverter;
    }

    @Override
    public List<InventoryAlert> findAll() {
        return alertConverter.toEntityList(alertMapper.selectList(null));
    }

    @Override
    public List<InventoryAlert> findActiveAlerts() {
        LambdaQueryWrapper<InventoryAlertDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.in(InventoryAlertDO::getStatus, 
                InventoryAlert.AlertStatus.TRIGGERED.name(),
                InventoryAlert.AlertStatus.ACKNOWLEDGED.name(),
                InventoryAlert.AlertStatus.IN_PROGRESS.name());
        return alertConverter.toEntityList(alertMapper.selectList(wrapper));
    }

    @Override
    public InventoryAlert findById(Long id) {
        InventoryAlertDO alertDO = alertMapper.selectById(id);
        return alertConverter.toEntity(alertDO);
    }

    @Override
    public InventoryAlert findByAlertCode(String alertCode) {
        LambdaQueryWrapper<InventoryAlertDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InventoryAlertDO::getAlertCode, alertCode);
        InventoryAlertDO alertDO = alertMapper.selectOne(wrapper);
        return alertConverter.toEntity(alertDO);
    }

    @Override
    public InventoryAlert findLastBySkuCode(String skuCode) {
        LambdaQueryWrapper<InventoryAlertDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InventoryAlertDO::getSkuCode, skuCode)
                .orderByDesc(InventoryAlertDO::getCreatedAt)
                .last("LIMIT 1");
        InventoryAlertDO alertDO = alertMapper.selectOne(wrapper);
        return alertConverter.toEntity(alertDO);
    }

    @Override
    public List<InventoryAlert> findBySkuCodeAndStatus(String skuCode, InventoryAlert.AlertStatus status) {
        LambdaQueryWrapper<InventoryAlertDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InventoryAlertDO::getSkuCode, skuCode);
        if (status != null) {
            wrapper.eq(InventoryAlertDO::getStatus, status.name());
        }
        return alertConverter.toEntityList(alertMapper.selectList(wrapper));
    }

    @Override
    public InventoryAlert save(InventoryAlert alert) {
        InventoryAlertDO alertDO = alertConverter.toDO(alert);
        if (alert.getId() == null) {
            alertMapper.insert(alertDO);
            alert.setId(alertDO.getId());
        } else {
            alertMapper.updateById(alertDO);
        }
        return alert;
    }

    @Override
    public void deleteById(Long id) {
        alertMapper.deleteById(id);
    }

    @Override
    public long countBySkuCodeAndStatus(String skuCode, InventoryAlert.AlertStatus status) {
        LambdaQueryWrapper<InventoryAlertDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InventoryAlertDO::getSkuCode, skuCode)
                .eq(InventoryAlertDO::getStatus, status.name());
        return alertMapper.selectCount(wrapper);
    }
    
    @Override
    public List<InventoryAlert> findLastBySkuCodes(List<String> skuCodes) {
        if (skuCodes == null || skuCodes.isEmpty()) {
            return List.of();
        }
        // 获取所有 SKU 的最新预警，按 SKU 和创建时间倒序
        LambdaQueryWrapper<InventoryAlertDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.in(InventoryAlertDO::getSkuCode, skuCodes)
                .orderByDesc(InventoryAlertDO::getSkuCode)
                .orderByDesc(InventoryAlertDO::getCreatedAt);
        List<InventoryAlertDO> allAlerts = alertMapper.selectList(wrapper);
        
        // 按 SKU 分组，取每个 SKU 的最新一条
        Map<String, InventoryAlertDO> latestBySku = new LinkedHashMap<>();
        for (InventoryAlertDO alert : allAlerts) {
            latestBySku.putIfAbsent(alert.getSkuCode(), alert);
        }
        return alertConverter.toEntityList(new ArrayList<>(latestBySku.values()));
    }
}