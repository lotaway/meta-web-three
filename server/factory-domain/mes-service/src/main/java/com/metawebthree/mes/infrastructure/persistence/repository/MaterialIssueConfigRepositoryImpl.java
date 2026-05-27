package com.metawebthree.mes.infrastructure.persistence.repository;

import com.metawebthree.mes.domain.entity.MaterialIssueConfig;
import com.metawebthree.mes.domain.repository.MaterialIssueConfigRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.MaterialIssueConfigDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.MaterialIssueConfigMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public class MaterialIssueConfigRepositoryImpl implements MaterialIssueConfigRepository {
    
    private final MaterialIssueConfigMapper mapper;
    
    public MaterialIssueConfigRepositoryImpl(MaterialIssueConfigMapper mapper) {
        this.mapper = mapper;
    }
    
    @Override
    public Optional<MaterialIssueConfig> findById(Long id) {
        MaterialIssueConfigDO obj = mapper.selectById(id);
        return Optional.ofNullable(toEntity(obj));
    }
    
    @Override
    public MaterialIssueConfig save(MaterialIssueConfig config) {
        if (config.getId() == null) {
            mapper.insert(toDO(config));
        } else {
            mapper.updateById(toDO(config));
        }
        return config;
    }
    
    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }
    
    @Override
    public Optional<MaterialIssueConfig> findActiveByWorkshopAndProduct(String workshopId, String productCode) {
        MaterialIssueConfigDO obj = mapper.findActiveByWorkshopAndProduct(workshopId, productCode);
        return Optional.ofNullable(toEntity(obj));
    }
    
    @Override
    public Optional<MaterialIssueConfig> findActiveByWorkshopDefault(String workshopId) {
        MaterialIssueConfigDO obj = mapper.findActiveByWorkshopDefault(workshopId);
        return Optional.ofNullable(toEntity(obj));
    }
    
    @Override
    public List<MaterialIssueConfig> findAllActive() {
        return mapper.findAllActive().stream()
                .map(this::toEntity)
                .toList();
    }
    
    @Override
    public Optional<MaterialIssueConfig> findActiveByConfigCode(String configCode) {
        MaterialIssueConfigDO obj = mapper.findActiveByConfigCode(configCode);
        return Optional.ofNullable(toEntity(obj));
    }
    
    @Override
    public List<MaterialIssueConfig> findActiveByIssueMode(String issueMode) {
        return mapper.findActiveByIssueMode(issueMode).stream()
                .map(this::toEntity)
                .toList();
    }
    
    private MaterialIssueConfig toEntity(MaterialIssueConfigDO obj) {
        if (obj == null) return null;
        MaterialIssueConfig entity = new MaterialIssueConfig();
        entity.setId(obj.getId());
        entity.setConfigCode(obj.getConfigCode());
        entity.setConfigName(obj.getConfigName());
        entity.setWorkshopId(obj.getWorkshopId());
        entity.setProductCode(obj.getProductCode());
        entity.setIssueMode(obj.getIssueMode());
        entity.setIssueRule(obj.getIssueRule());
        entity.setLeadTimeHours(obj.getLeadTimeHours());
        entity.setBufferHours(obj.getBufferHours());
        entity.setIsActive(obj.getIsActive());
        entity.setPriority(obj.getPriority());
        entity.setDescription(obj.getDescription());
        entity.setCreatedAt(obj.getCreatedAt());
        entity.setUpdatedAt(obj.getUpdatedAt());
        return entity;
    }
    
    private MaterialIssueConfigDO toDO(MaterialIssueConfig entity) {
        return MaterialIssueConfigDO.builder()
                .id(entity.getId())
                .configCode(entity.getConfigCode())
                .configName(entity.getConfigName())
                .workshopId(entity.getWorkshopId())
                .productCode(entity.getProductCode())
                .issueMode(entity.getIssueMode())
                .issueRule(entity.getIssueRule())
                .leadTimeHours(entity.getLeadTimeHours())
                .bufferHours(entity.getBufferHours())
                .isActive(entity.getIsActive())
                .priority(entity.getPriority())
                .description(entity.getDescription())
                .createdAt(entity.getCreatedAt())
                .updatedAt(entity.getUpdatedAt())
                .build();
    }
}