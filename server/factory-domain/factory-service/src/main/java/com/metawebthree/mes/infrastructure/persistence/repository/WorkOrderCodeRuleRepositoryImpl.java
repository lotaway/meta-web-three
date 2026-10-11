package com.metawebthree.mes.infrastructure.persistence.repository;

import com.metawebthree.mes.domain.entity.WorkOrderCodeRule;
import com.metawebthree.mes.domain.repository.WorkOrderCodeRuleRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.WorkOrderCodeRuleDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.WorkOrderCodeRuleMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public class WorkOrderCodeRuleRepositoryImpl implements WorkOrderCodeRuleRepository {
    
    private final WorkOrderCodeRuleMapper mapper;
    
    public WorkOrderCodeRuleRepositoryImpl(WorkOrderCodeRuleMapper mapper) {
        this.mapper = mapper;
    }
    
    @Override
    public Optional<WorkOrderCodeRule> findById(Long id) {
        WorkOrderCodeRuleDO obj = mapper.selectById(id);
        return Optional.ofNullable(toEntity(obj));
    }
    
    @Override
    public WorkOrderCodeRule save(WorkOrderCodeRule binding) {
        if (binding.getId() == null) {
            mapper.insert(toDO(binding));
        } else {
            mapper.updateById(toDO(binding));
        }
        return binding;
    }
    
    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }
    
    @Override
    public Optional<WorkOrderCodeRule> findActiveByWorkshopAndType(String workshopId, String workOrderType) {
        WorkOrderCodeRuleDO obj = mapper.findActiveByWorkshopAndType(workshopId, workOrderType);
        return Optional.ofNullable(toEntity(obj));
    }
    
    @Override
    public Optional<WorkOrderCodeRule> findActiveByType(String workOrderType) {
        WorkOrderCodeRuleDO obj = mapper.findActiveByType(workOrderType);
        return Optional.ofNullable(toEntity(obj));
    }
    
    @Override
    public Optional<WorkOrderCodeRule> findActiveByWorkshop(String workshopId) {
        WorkOrderCodeRuleDO obj = mapper.findActiveByWorkshop(workshopId);
        return Optional.ofNullable(toEntity(obj));
    }
    
    @Override
    public List<WorkOrderCodeRule> findAllActive() {
        return mapper.findAllActive().stream()
                .map(this::toEntity)
                .toList();
    }
    
    @Override
    public Optional<WorkOrderCodeRule> findActiveByRuleCode(String ruleCode) {
        WorkOrderCodeRuleDO obj = mapper.findActiveByRuleCode(ruleCode);
        return Optional.ofNullable(toEntity(obj));
    }
    
    private WorkOrderCodeRule toEntity(WorkOrderCodeRuleDO obj) {
        if (obj == null) return null;
        WorkOrderCodeRule entity = new WorkOrderCodeRule();
        entity.setId(obj.getId());
        entity.setWorkshopId(obj.getWorkshopId());
        entity.setWorkOrderType(obj.getWorkOrderType());
        entity.setCodeRuleId(obj.getCodeRuleId());
        entity.setRuleCode(obj.getRuleCode());
        entity.setIsActive(obj.getIsActive());
        entity.setDescription(obj.getDescription());
        entity.setPriority(obj.getPriority());
        entity.setCreatedAt(obj.getCreatedAt());
        entity.setUpdatedAt(obj.getUpdatedAt());
        return entity;
    }
    
    private WorkOrderCodeRuleDO toDO(WorkOrderCodeRule entity) {
        return WorkOrderCodeRuleDO.builder()
                .id(entity.getId())
                .workshopId(entity.getWorkshopId())
                .workOrderType(entity.getWorkOrderType())
                .codeRuleId(entity.getCodeRuleId())
                .ruleCode(entity.getRuleCode())
                .isActive(entity.getIsActive())
                .description(entity.getDescription())
                .priority(entity.getPriority())
                .createdAt(entity.getCreatedAt())
                .updatedAt(entity.getUpdatedAt())
                .build();
    }
}