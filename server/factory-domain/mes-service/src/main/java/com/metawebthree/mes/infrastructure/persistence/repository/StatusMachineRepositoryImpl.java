package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.config.StatusMachine;
import com.metawebthree.mes.domain.config.StatusConfig;
import com.metawebthree.mes.domain.config.StatusTransitionRule;
import com.metawebthree.mes.domain.repository.StatusMachineRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.StatusMachineDO;
import com.metawebthree.mes.infrastructure.persistence.dataobject.StatusConfigDO;
import com.metawebthree.mes.infrastructure.persistence.dataobject.StatusTransitionDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.StatusMachineMapper;
import com.metawebthree.mes.infrastructure.persistence.mapper.StatusConfigMapper;
import com.metawebthree.mes.infrastructure.persistence.mapper.StatusTransitionMapper;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class StatusMachineRepositoryImpl implements StatusMachineRepository {
    
    private final StatusMachineMapper statusMachineMapper;
    private final StatusConfigMapper statusConfigMapper;
    private final StatusTransitionMapper statusTransitionMapper;
    
    public StatusMachineRepositoryImpl(
            StatusMachineMapper statusMachineMapper,
            StatusConfigMapper statusConfigMapper,
            StatusTransitionMapper statusTransitionMapper) {
        this.statusMachineMapper = statusMachineMapper;
        this.statusConfigMapper = statusConfigMapper;
        this.statusTransitionMapper = statusTransitionMapper;
    }
    
    @Override
    public Optional<StatusMachine> findById(Long id) {
        StatusMachineDO dto = statusMachineMapper.selectById(id);
        return Optional.ofNullable(dto).map(this::toDomain);
    }
    
    @Override
    public Optional<StatusMachine> findByMachineCode(String machineCode) {
        LambdaQueryWrapper<StatusMachineDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(StatusMachineDO::getMachineCode, machineCode);
        StatusMachineDO dto = statusMachineMapper.selectOne(wrapper);
        return Optional.ofNullable(dto).map(this::toDomain);
    }
    
    @Override
    public Optional<StatusMachine> findByEntityTypeAndIsDefaultTrue(String entityType) {
        LambdaQueryWrapper<StatusMachineDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(StatusMachineDO::getEntityType, entityType)
               .eq(StatusMachineDO::getIsDefault, true);
        StatusMachineDO dto = statusMachineMapper.selectOne(wrapper);
        return Optional.ofNullable(dto).map(this::toDomain);
    }
    
    @Override
    public Optional<StatusMachine> findByEntityType(String entityType) {
        // Try to find default first, otherwise any active one
        Optional<StatusMachine> defaultMachine = findByEntityTypeAndIsDefaultTrue(entityType);
        if (defaultMachine.isPresent()) {
            return defaultMachine;
        }
        
        LambdaQueryWrapper<StatusMachineDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(StatusMachineDO::getEntityType, entityType)
               .eq(StatusMachineDO::getStatus, "ACTIVE")
               .last("LIMIT 1");
        StatusMachineDO dto = statusMachineMapper.selectOne(wrapper);
        return Optional.ofNullable(dto).map(this::toDomain);
    }
    
    @Override
    public StatusMachine save(StatusMachine statusMachine) {
        StatusMachineDO dto = toDO(statusMachine);
        if (dto.getId() == null) {
            dto.setCreatedAt(LocalDateTime.now());
            dto.setUpdatedAt(LocalDateTime.now());
            statusMachineMapper.insert(dto);
        } else {
            dto.setUpdatedAt(LocalDateTime.now());
            statusMachineMapper.updateById(dto);
        }
        return toDomain(dto);
    }
    
    @Override
    public void update(StatusMachine statusMachine) {
        StatusMachineDO dto = toDO(statusMachine);
        dto.setUpdatedAt(LocalDateTime.now());
        statusMachineMapper.updateById(dto);
    }
    
    @Override
    public void deleteById(Long id) {
        statusMachineMapper.deleteById(id);
    }
    
    @Override
    public List<StatusConfig> findStatusesByMachineId(Long machineId) {
        LambdaQueryWrapper<StatusConfigDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(StatusConfigDO::getMachineId, machineId)
               .orderByAsc(StatusConfigDO::getSortOrder);
        return statusConfigMapper.selectList(wrapper).stream()
                .map(this::toStatusConfigDomain)
                .collect(Collectors.toList());
    }
    
    @Override
    public List<StatusTransitionRule> findTransitionsByMachineId(Long machineId) {
        LambdaQueryWrapper<StatusTransitionDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(StatusTransitionDO::getMachineId, machineId)
               .orderByAsc(StatusTransitionDO::getSortOrder);
        return statusTransitionMapper.selectList(wrapper).stream()
                .map(this::toTransitionDomain)
                .collect(Collectors.toList());
    }
    
    @Override
    public StatusConfig saveStatus(StatusConfig statusConfig) {
        StatusConfigDO dto = toStatusConfigDO(statusConfig);
        if (dto.getId() == null) {
            dto.setCreatedAt(LocalDateTime.now());
            statusConfigMapper.insert(dto);
        } else {
            statusConfigMapper.updateById(dto);
        }
        return toStatusConfigDomain(dto);
    }
    
    @Override
    public void updateStatus(StatusConfig statusConfig) {
        StatusConfigDO dto = toStatusConfigDO(statusConfig);
        statusConfigMapper.updateById(dto);
    }
    
    @Override
    public void deleteStatusById(Long id) {
        statusConfigMapper.deleteById(id);
    }
    
    @Override
    public StatusTransitionRule saveTransition(StatusTransitionRule transition) {
        StatusTransitionDO dto = toTransitionDO(transition);
        if (dto.getId() == null) {
            dto.setCreatedAt(LocalDateTime.now());
            statusTransitionMapper.insert(dto);
        } else {
            statusTransitionMapper.updateById(dto);
        }
        return toTransitionDomain(dto);
    }
    
    @Override
    public void updateTransition(StatusTransitionRule transition) {
        StatusTransitionDO dto = toTransitionDO(transition);
        statusTransitionMapper.updateById(dto);
    }
    
    @Override
    public void deleteTransitionById(Long id) {
        statusTransitionMapper.deleteById(id);
    }
    
    private StatusMachine toDomain(StatusMachineDO dto) {
        if (dto == null) return null;
        StatusMachine domain = new StatusMachine();
        domain.setId(dto.getId());
        domain.setMachineCode(dto.getMachineCode());
        domain.setMachineName(dto.getMachineName());
        domain.setEntityType(dto.getEntityType());
        domain.setDescription(dto.getDescription());
        domain.setInitialStatus(dto.getInitialStatus());
        domain.setIsDefault(dto.getIsDefault());
        domain.setStatus(dto.getStatus());
        
        // Load related statuses and transitions
        domain.setStatuses(findStatusesByMachineId(dto.getId()));
        domain.setTransitions(findTransitionsByMachineId(dto.getId()));
        
        return domain;
    }
    
    private StatusMachineDO toDO(StatusMachine domain) {
        if (domain == null) return null;
        StatusMachineDO dto = new StatusMachineDO();
        dto.setId(domain.getId());
        dto.setMachineCode(domain.getMachineCode());
        dto.setMachineName(domain.getMachineName());
        dto.setEntityType(domain.getEntityType());
        dto.setDescription(domain.getDescription());
        dto.setInitialStatus(domain.getInitialStatus());
        dto.setIsDefault(domain.getIsDefault());
        dto.setStatus(domain.getStatus());
        return dto;
    }
    
    private StatusConfig toStatusConfigDomain(StatusConfigDO dto) {
        if (dto == null) return null;
        StatusConfig domain = new StatusConfig();
        domain.setId(dto.getId());
        domain.setMachineId(dto.getMachineId());
        domain.setStatusCode(dto.getStatusCode());
        domain.setStatusName(dto.getStatusName());
        domain.setStatusCategory(dto.getStatusCategory());
        domain.setIsInitial(dto.getIsInitial());
        domain.setIsFinal(dto.getIsFinal());
        domain.setColor(dto.getColor());
        domain.setIcon(dto.getIcon());
        domain.setSortOrder(dto.getSortOrder());
        return domain;
    }
    
    private StatusConfigDO toStatusConfigDO(StatusConfig domain) {
        if (domain == null) return null;
        StatusConfigDO dto = new StatusConfigDO();
        dto.setId(domain.getId());
        dto.setMachineId(domain.getMachineId());
        dto.setStatusCode(domain.getStatusCode());
        dto.setStatusName(domain.getStatusName());
        dto.setStatusCategory(domain.getStatusCategory());
        dto.setIsInitial(domain.getIsInitial());
        dto.setIsFinal(domain.getIsFinal());
        dto.setColor(domain.getColor());
        dto.setIcon(domain.getIcon());
        dto.setSortOrder(domain.getSortOrder());
        return dto;
    }
    
    private StatusTransitionRule toTransitionDomain(StatusTransitionDO dto) {
        if (dto == null) return null;
        StatusTransitionRule domain = new StatusTransitionRule();
        domain.setId(dto.getId());
        domain.setMachineId(dto.getMachineId());
        domain.setFromStatus(dto.getFromStatus());
        domain.setToStatus(dto.getToStatus());
        domain.setTransitionAction(dto.getTransitionAction());
        domain.setConditionExpression(dto.getConditionExpression());
        domain.setEventCode(dto.getEventCode());
        domain.setIsAutoTransition(dto.getIsAutoTransition());
        domain.setSortOrder(dto.getSortOrder());
        return domain;
    }
    
    private StatusTransitionDO toTransitionDO(StatusTransitionRule domain) {
        if (domain == null) return null;
        StatusTransitionDO dto = new StatusTransitionDO();
        dto.setId(domain.getId());
        dto.setMachineId(domain.getMachineId());
        dto.setFromStatus(domain.getFromStatus());
        dto.setToStatus(domain.getToStatus());
        dto.setTransitionAction(domain.getTransitionAction());
        dto.setConditionExpression(domain.getConditionExpression());
        dto.setEventCode(domain.getEventCode());
        dto.setIsAutoTransition(domain.getIsAutoTransition());
        dto.setSortOrder(domain.getSortOrder());
        return dto;
    }
}