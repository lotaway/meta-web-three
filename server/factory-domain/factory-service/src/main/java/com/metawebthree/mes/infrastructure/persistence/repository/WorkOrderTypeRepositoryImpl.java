package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.config.WorkOrderType;
import com.metawebthree.mes.domain.repository.WorkOrderTypeRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.WorkOrderTypeDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.WorkOrderTypeMapper;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class WorkOrderTypeRepositoryImpl implements WorkOrderTypeRepository {
    
    private final WorkOrderTypeMapper workOrderTypeMapper;
    
    public WorkOrderTypeRepositoryImpl(WorkOrderTypeMapper workOrderTypeMapper) {
        this.workOrderTypeMapper = workOrderTypeMapper;
    }
    
    @Override
    public Optional<WorkOrderType> findById(Long id) {
        WorkOrderTypeDO dto = workOrderTypeMapper.selectById(id);
        return Optional.ofNullable(dto).map(this::toDomain);
    }
    
    @Override
    public Optional<WorkOrderType> findByTypeCode(String typeCode) {
        LambdaQueryWrapper<WorkOrderTypeDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkOrderTypeDO::getTypeCode, typeCode);
        WorkOrderTypeDO dto = workOrderTypeMapper.selectOne(wrapper);
        return Optional.ofNullable(dto).map(this::toDomain);
    }
    
    @Override
    public Optional<WorkOrderType> findByIsDefaultTrue() {
        LambdaQueryWrapper<WorkOrderTypeDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkOrderTypeDO::getIsDefault, true);
        WorkOrderTypeDO dto = workOrderTypeMapper.selectOne(wrapper);
        return Optional.ofNullable(dto).map(this::toDomain);
    }
    
    @Override
    public List<WorkOrderType> findAll() {
        return workOrderTypeMapper.selectList(null).stream()
                .map(this::toDomain)
                .collect(Collectors.toList());
    }
    
    @Override
    public List<WorkOrderType> findByStatus(String status) {
        LambdaQueryWrapper<WorkOrderTypeDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkOrderTypeDO::getStatus, status);
        return workOrderTypeMapper.selectList(wrapper).stream()
                .map(this::toDomain)
                .collect(Collectors.toList());
    }
    
    @Override
    public WorkOrderType save(WorkOrderType workOrderType) {
        WorkOrderTypeDO dto = toDO(workOrderType);
        if (dto.getId() == null) {
            dto.setCreatedAt(LocalDateTime.now());
            dto.setUpdatedAt(LocalDateTime.now());
            workOrderTypeMapper.insert(dto);
        } else {
            dto.setUpdatedAt(LocalDateTime.now());
            workOrderTypeMapper.updateById(dto);
        }
        return toDomain(dto);
    }
    
    @Override
    public void update(WorkOrderType workOrderType) {
        WorkOrderTypeDO dto = toDO(workOrderType);
        dto.setUpdatedAt(LocalDateTime.now());
        workOrderTypeMapper.updateById(dto);
    }
    
    @Override
    public void deleteById(Long id) {
        workOrderTypeMapper.deleteById(id);
    }
    
    private WorkOrderType toDomain(WorkOrderTypeDO dto) {
        if (dto == null) return null;
        WorkOrderType domain = new WorkOrderType();
        domain.setId(dto.getId());
        domain.setTypeCode(dto.getTypeCode());
        domain.setTypeName(dto.getTypeName());
        domain.setDescription(dto.getDescription());
        domain.setStatusMachineCode(dto.getStatusMachineCode());
        domain.setProcessRouteTemplate(dto.getProcessRouteTemplate());
        domain.setIsDefault(dto.getIsDefault());
        domain.setSortOrder(dto.getSortOrder());
        domain.setStatus(dto.getStatus());
        return domain;
    }
    
    private WorkOrderTypeDO toDO(WorkOrderType domain) {
        if (domain == null) return null;
        WorkOrderTypeDO dto = new WorkOrderTypeDO();
        dto.setId(domain.getId());
        dto.setTypeCode(domain.getTypeCode());
        dto.setTypeName(domain.getTypeName());
        dto.setDescription(domain.getDescription());
        dto.setStatusMachineCode(domain.getStatusMachineCode());
        dto.setProcessRouteTemplate(domain.getProcessRouteTemplate());
        dto.setIsDefault(domain.getIsDefault());
        dto.setSortOrder(domain.getSortOrder());
        dto.setStatus(domain.getStatus());
        return dto;
    }
}