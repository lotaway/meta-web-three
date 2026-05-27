package com.metawebthree.production.infrastructure.persistence.repository;

import com.metawebthree.production.domain.entity.WorkStationBinding;
import com.metawebthree.production.domain.repository.WorkStationBindingRepository;
import com.metawebthree.production.infrastructure.persistence.dataobject.WorkStationBindingDO;
import com.metawebthree.production.infrastructure.persistence.mapper.WorkStationBindingMapper;
import org.springframework.stereotype.Repository;
import java.util.List;
import java.util.stream.Collectors;

@Repository
public class WorkStationBindingRepositoryImpl implements WorkStationBindingRepository {
    
    private final WorkStationBindingMapper mapper;
    
    public WorkStationBindingRepositoryImpl(WorkStationBindingMapper mapper) {
        this.mapper = mapper;
    }
    
    @Override
    public void save(WorkStationBinding binding) {
        WorkStationBindingDO dataObject = toDO(binding);
        mapper.insert(dataObject);
        binding.setId(dataObject.getId());
    }
    
    @Override
    public void update(WorkStationBinding binding) {
        mapper.updateById(toDO(binding));
    }
    
    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }
    
    @Override
    public WorkStationBinding findById(Long id) {
        WorkStationBindingDO dataObject = mapper.selectById(id);
        return dataObject == null ? null : toEntity(dataObject);
    }
    
    @Override
    public List<WorkStationBinding> findByWorkstationCode(String workstationCode) {
        List<WorkStationBindingDO> list = mapper.selectList(
            new com.baomidou.mybatisplus.core.conditions.query.QueryWrapper<WorkStationBindingDO>()
                .eq("workstation_code", workstationCode)
                .eq("status", "ACTIVE")
        );
        return list.stream().map(this::toEntity).collect(Collectors.toList());
    }
    
    @Override
    public List<WorkStationBinding> findByWorkstationCodeAndType(String workstationCode, 
                                                                  WorkStationBinding.BindingType type) {
        List<WorkStationBindingDO> list = mapper.selectList(
            new com.baomidou.mybatisplus.core.conditions.query.QueryWrapper<WorkStationBindingDO>()
                .eq("workstation_code", workstationCode)
                .eq("binding_type", type.name())
                .eq("status", "ACTIVE")
        );
        return list.stream().map(this::toEntity).collect(Collectors.toList());
    }
    
    @Override
    public List<WorkStationBinding> findByTargetCode(String targetCode) {
        List<WorkStationBindingDO> list = mapper.selectList(
            new com.baomidou.mybatisplus.core.conditions.query.QueryWrapper<WorkStationBindingDO>()
                .eq("target_code", targetCode)
                .eq("status", "ACTIVE")
        );
        return list.stream().map(this::toEntity).collect(Collectors.toList());
    }
    
    @Override
    public WorkStationBinding findPrimaryByWorkstationAndType(String workstationCode, 
                                                               WorkStationBinding.BindingType type) {
        WorkStationBindingDO dataObject = mapper.selectOne(
            new com.baomidou.mybatisplus.core.conditions.query.QueryWrapper<WorkStationBindingDO>()
                .eq("workstation_code", workstationCode)
                .eq("binding_type", type.name())
                .eq("is_primary", true)
                .eq("status", "ACTIVE")
        );
        return dataObject == null ? null : toEntity(dataObject);
    }
    
    private WorkStationBindingDO toDO(WorkStationBinding entity) {
        WorkStationBindingDO dataObject = new WorkStationBindingDO();
        dataObject.setId(entity.getId());
        dataObject.setWorkstationCode(entity.getWorkstationCode());
        dataObject.setBindingType(entity.getBindingType() == null ? null : entity.getBindingType().name());
        dataObject.setTargetCode(entity.getTargetCode());
        dataObject.setTargetName(entity.getTargetName());
        dataObject.setTargetType(entity.getTargetType());
        dataObject.setQuantity(entity.getQuantity());
        dataObject.setIsPrimary(entity.getIsPrimary());
        dataObject.setStatus(entity.getStatus().name());
        dataObject.setRemark(entity.getRemark());
        dataObject.setCreatedAt(entity.getCreatedAt());
        dataObject.setUpdatedAt(entity.getUpdatedAt());
        return dataObject;
    }
    
    private WorkStationBinding toEntity(WorkStationBindingDO dataObject) {
        WorkStationBinding entity = new WorkStationBinding();
        entity.setId(dataObject.getId());
        entity.setWorkstationCode(dataObject.getWorkstationCode());
        entity.setBindingType(dataObject.getBindingType() == null ? null : 
            WorkStationBinding.BindingType.valueOf(dataObject.getBindingType()));
        entity.setTargetCode(dataObject.getTargetCode());
        entity.setTargetName(dataObject.getTargetName());
        entity.setTargetType(dataObject.getTargetType());
        entity.setQuantity(dataObject.getQuantity());
        entity.setIsPrimary(dataObject.getIsPrimary());
        entity.setStatus(WorkStationBinding.Status.valueOf(dataObject.getStatus()));
        entity.setRemark(dataObject.getRemark());
        entity.setCreatedAt(dataObject.getCreatedAt());
        entity.setUpdatedAt(dataObject.getUpdatedAt());
        return entity;
    }
}