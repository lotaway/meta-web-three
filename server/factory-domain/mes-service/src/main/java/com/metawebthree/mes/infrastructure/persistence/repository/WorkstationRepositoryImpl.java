package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.Workstation;
import com.metawebthree.mes.domain.repository.WorkstationRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.WorkstationDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.WorkstationMapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public class WorkstationRepositoryImpl implements WorkstationRepository {
    
    @Autowired
    private WorkstationMapper workstationMapper;
    
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    @Override
    public Optional<Workstation> findById(Long id) {
        WorkstationDO workstationDO = workstationMapper.selectById(id);
        return Optional.ofNullable(workstationDO).map(this::toEntity);
    }
    
    @Override
    public Optional<Workstation> findByWorkstationCode(String workstationCode) {
        LambdaQueryWrapper<WorkstationDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkstationDO::getWorkstationCode, workstationCode);
        WorkstationDO workstationDO = workstationMapper.selectOne(wrapper);
        return Optional.ofNullable(workstationDO).map(this::toEntity);
    }
    
    @Override
    public List<Workstation> findByWorkshopId(String workshopId) {
        LambdaQueryWrapper<WorkstationDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkstationDO::getWorkshopId, workshopId);
        List<WorkstationDO> doList = workstationMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<Workstation> findByStatus(Workstation.WorkstationStatus status) {
        LambdaQueryWrapper<WorkstationDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkstationDO::getStatus, status.name());
        List<WorkstationDO> doList = workstationMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<Workstation> findByType(Workstation.WorkstationType type) {
        LambdaQueryWrapper<WorkstationDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkstationDO::getType, type.name());
        List<WorkstationDO> doList = workstationMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public Workstation save(Workstation workstation) {
        WorkstationDO workstationDO = toDO(workstation);
        if (workstation.getId() == null) {
            workstationMapper.insert(workstationDO);
            workstation.setId(workstationDO.getId());
        } else {
            workstationMapper.updateById(workstationDO);
        }
        return workstation;
    }
    
    @Override
    public void update(Workstation workstation) {
        if (workstation.getId() != null) {
            WorkstationDO workstationDO = toDO(workstation);
            workstationMapper.updateById(workstationDO);
        }
    }
    
    @Override
    public void deleteById(Long id) {
        workstationMapper.deleteById(id);
    }
    
    private Workstation toEntity(WorkstationDO doObj) {
        if (doObj == null) {
            return null;
        }
        Workstation entity = new Workstation();
        entity.setId(doObj.getId());
        entity.setWorkstationCode(doObj.getWorkstationCode());
        entity.setWorkstationName(doObj.getWorkstationName());
        entity.setWorkshopId(doObj.getWorkshopId());
        entity.setWorkshopName(doObj.getWorkshopName());
        entity.setType(Workstation.WorkstationType.valueOf(doObj.getType()));
        entity.setStatus(Workstation.WorkstationStatus.valueOf(doObj.getStatus()));
        entity.setLocation(doObj.getLocation());
        entity.setCapacity(doObj.getCapacity());
        entity.setDescription(doObj.getDescription());
        
        // Parse JSON arrays
        if (doObj.getEquipmentIds() != null && !doObj.getEquipmentIds().isEmpty()) {
            try {
                entity.setEquipmentIds(objectMapper.readValue(doObj.getEquipmentIds(), 
                    new TypeReference<List<Long>>() {}));
            } catch (JsonProcessingException e) {
                // ignore
            }
        }
        if (doObj.getEquipmentCodes() != null && !doObj.getEquipmentCodes().isEmpty()) {
            try {
                entity.setEquipmentCodes(objectMapper.readValue(doObj.getEquipmentCodes(), 
                    new TypeReference<List<String>>() {}));
            } catch (JsonProcessingException e) {
                // ignore
            }
        }
        if (doObj.getToolIds() != null && !doObj.getToolIds().isEmpty()) {
            try {
                entity.setToolIds(objectMapper.readValue(doObj.getToolIds(), 
                    new TypeReference<List<Long>>() {}));
            } catch (JsonProcessingException e) {
                // ignore
            }
        }
        if (doObj.getToolNames() != null && !doObj.getToolNames().isEmpty()) {
            try {
                entity.setToolNames(objectMapper.readValue(doObj.getToolNames(), 
                    new TypeReference<List<String>>() {}));
            } catch (JsonProcessingException e) {
                // ignore
            }
        }
        if (doObj.getOperatorIds() != null && !doObj.getOperatorIds().isEmpty()) {
            try {
                entity.setOperatorIds(objectMapper.readValue(doObj.getOperatorIds(), 
                    new TypeReference<List<String>>() {}));
            } catch (JsonProcessingException e) {
                // ignore
            }
        }
        if (doObj.getOperatorNames() != null && !doObj.getOperatorNames().isEmpty()) {
            try {
                entity.setOperatorNames(objectMapper.readValue(doObj.getOperatorNames(), 
                    new TypeReference<List<String>>() {}));
            } catch (JsonProcessingException e) {
                // ignore
            }
        }
        
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }
    
    private WorkstationDO toDO(Workstation entity) {
        if (entity == null) {
            return null;
        }
        WorkstationDO doObj = new WorkstationDO();
        doObj.setId(entity.getId());
        doObj.setWorkstationCode(entity.getWorkstationCode());
        doObj.setWorkstationName(entity.getWorkstationName());
        doObj.setWorkshopId(entity.getWorkshopId());
        doObj.setWorkshopName(entity.getWorkshopName());
        doObj.setType(entity.getType() != null ? entity.getType().name() : null);
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        doObj.setLocation(entity.getLocation());
        doObj.setCapacity(entity.getCapacity());
        doObj.setDescription(entity.getDescription());
        
        // Convert lists to JSON
        try {
            if (entity.getEquipmentIds() != null) {
                doObj.setEquipmentIds(objectMapper.writeValueAsString(entity.getEquipmentIds()));
            }
            if (entity.getEquipmentCodes() != null) {
                doObj.setEquipmentCodes(objectMapper.writeValueAsString(entity.getEquipmentCodes()));
            }
            if (entity.getToolIds() != null) {
                doObj.setToolIds(objectMapper.writeValueAsString(entity.getToolIds()));
            }
            if (entity.getToolNames() != null) {
                doObj.setToolNames(objectMapper.writeValueAsString(entity.getToolNames()));
            }
            if (entity.getOperatorIds() != null) {
                doObj.setOperatorIds(objectMapper.writeValueAsString(entity.getOperatorIds()));
            }
            if (entity.getOperatorNames() != null) {
                doObj.setOperatorNames(objectMapper.writeValueAsString(entity.getOperatorNames()));
            }
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Failed to serialize workstation relations", e);
        }
        
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}