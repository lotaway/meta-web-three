package com.metawebthree.inventory.infrastructure.persistence.repository.alert;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.inventory.domain.entity.alert.InventoryAlertConfig;
import com.metawebthree.inventory.domain.repository.alert.InventoryAlertConfigRepository;
import com.metawebthree.inventory.infrastructure.persistence.converter.InventoryAlertConfigConverter;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.InventoryAlertConfigDO;
import com.metawebthree.inventory.infrastructure.persistence.mapper.InventoryAlertConfigMapper;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public class InventoryAlertConfigRepositoryImpl implements InventoryAlertConfigRepository {

    private final InventoryAlertConfigMapper configMapper;
    private final InventoryAlertConfigConverter configConverter;

    public InventoryAlertConfigRepositoryImpl(InventoryAlertConfigMapper configMapper, 
                                               InventoryAlertConfigConverter configConverter) {
        this.configMapper = configMapper;
        this.configConverter = configConverter;
    }

    @Override
    public List<InventoryAlertConfig> findAll() {
        return configConverter.toEntityList(configMapper.selectList(null));
    }

    @Override
    public List<InventoryAlertConfig> findAllEnabled() {
        LambdaQueryWrapper<InventoryAlertConfigDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(InventoryAlertConfigDO::getEnabled, true);
        return configConverter.toEntityList(configMapper.selectList(wrapper));
    }

    @Override
    public InventoryAlertConfig findById(Long id) {
        InventoryAlertConfigDO configDO = configMapper.selectById(id);
        return configConverter.toEntity(configDO);
    }

    @Override
    public InventoryAlertConfig save(InventoryAlertConfig config) {
        InventoryAlertConfigDO configDO = configConverter.toDO(config);
        if (config.getId() == null) {
            configMapper.insert(configDO);
            config.setId(configDO.getId());
        } else {
            configMapper.updateById(configDO);
        }
        return config;
    }

    @Override
    public void deleteById(Long id) {
        configMapper.deleteById(id);
    }
}