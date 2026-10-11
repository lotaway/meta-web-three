package com.metawebthree.recommendation.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.recommendation.domain.aishopping.entity.AiShoppingConfig;
import com.metawebthree.recommendation.domain.aishopping.repository.AiShoppingConfigRepository;
import com.metawebthree.recommendation.infrastructure.persistence.entity.AiShoppingConfigDO;
import com.metawebthree.recommendation.infrastructure.persistence.mapper.AiShoppingConfigMapper;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import org.springframework.stereotype.Repository;

@Repository
public class AiShoppingConfigRepositoryImpl implements AiShoppingConfigRepository {

    private final AiShoppingConfigMapper configMapper;

    public AiShoppingConfigRepositoryImpl(AiShoppingConfigMapper configMapper) {
        this.configMapper = configMapper;
    }

    @Override
    public Optional<String> findValue(String configKey) {
        AiShoppingConfigDO entity = configMapper.selectById(configKey);
        return entity != null && entity.getConfigValue() != null
                ? Optional.of(entity.getConfigValue())
                : Optional.empty();
    }

    @Override
    public Optional<AiShoppingConfig> findById(String configKey) {
        AiShoppingConfigDO entity = configMapper.selectById(configKey);
        return Optional.ofNullable(toDomain(entity));
    }

    @Override
    public AiShoppingConfig save(AiShoppingConfig config) {
        AiShoppingConfigDO entity = toDO(config);
        if (configMapper.selectById(entity.getConfigKey()) != null) {
            entity.setUpdatedAt(LocalDateTime.now());
            configMapper.updateById(entity);
        } else {
            entity.setUpdatedAt(LocalDateTime.now());
            configMapper.insert(entity);
        }
        config.setUpdatedAt(entity.getUpdatedAt());
        return config;
    }

    @Override
    public List<AiShoppingConfig> findAll() {
        return configMapper.selectList(new LambdaQueryWrapper<AiShoppingConfigDO>()
                        .orderByAsc(AiShoppingConfigDO::getConfigKey))
                .stream()
                .map(this::toDomain)
                .collect(Collectors.toList());
    }

    @Override
    public void delete(String configKey) {
        configMapper.deleteById(configKey);
    }

    private AiShoppingConfig toDomain(AiShoppingConfigDO entity) {
        if (entity == null) {
            return null;
        }
        AiShoppingConfig config = new AiShoppingConfig();
        config.setConfigKey(entity.getConfigKey());
        config.setConfigValue(entity.getConfigValue());
        config.setDescription(entity.getDescription());
        config.setUpdatedAt(entity.getUpdatedAt());
        return config;
    }

    private AiShoppingConfigDO toDO(AiShoppingConfig config) {
        AiShoppingConfigDO entity = new AiShoppingConfigDO();
        entity.setConfigKey(config.getConfigKey());
        entity.setConfigValue(config.getConfigValue());
        entity.setDescription(config.getDescription());
        entity.setUpdatedAt(config.getUpdatedAt());
        return entity;
    }
}
