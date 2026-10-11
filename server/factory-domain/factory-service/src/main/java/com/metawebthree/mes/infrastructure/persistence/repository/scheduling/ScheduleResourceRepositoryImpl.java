package com.metawebthree.mes.infrastructure.persistence.repository.scheduling;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.scheduling.ScheduleResource;
import com.metawebthree.mes.domain.entity.scheduling.ScheduleResource.ResourceType;
import com.metawebthree.mes.domain.entity.scheduling.ScheduleResource.ResourceStatus;
import com.metawebthree.mes.domain.repository.scheduling.ScheduleResourceRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.scheduling.ScheduleResourceDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.scheduling.ScheduleResourceMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class ScheduleResourceRepositoryImpl implements ScheduleResourceRepository {

    private final ScheduleResourceMapper resourceMapper;

    public ScheduleResourceRepositoryImpl(ScheduleResourceMapper resourceMapper) {
        this.resourceMapper = resourceMapper;
    }

    @Override
    public Optional<ScheduleResource> findById(Long id) {
        ScheduleResourceDO doObj = resourceMapper.selectById(id);
        return Optional.ofNullable(doObj).map(this::toEntity);
    }

    @Override
    public Optional<ScheduleResource> findByResourceCode(String resourceCode) {
        LambdaQueryWrapper<ScheduleResourceDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ScheduleResourceDO::getResourceCode, resourceCode);
        ScheduleResourceDO doObj = resourceMapper.selectOne(wrapper);
        return Optional.ofNullable(doObj).map(this::toEntity);
    }

    @Override
    public List<ScheduleResource> findByWorkshopId(String workshopId) {
        LambdaQueryWrapper<ScheduleResourceDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ScheduleResourceDO::getWorkshopId, workshopId);
        List<ScheduleResourceDO> doList = resourceMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<ScheduleResource> findByResourceType(ScheduleResource.ResourceType resourceType) {
        LambdaQueryWrapper<ScheduleResourceDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ScheduleResourceDO::getResourceType, resourceType.name());
        List<ScheduleResourceDO> doList = resourceMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<ScheduleResource> findAll() {
        List<ScheduleResourceDO> doList = resourceMapper.selectList(null);
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public ScheduleResource save(ScheduleResource entity) {
        ScheduleResourceDO doObj = toDO(entity);
        if (doObj.getId() == null) {
            resourceMapper.insert(doObj);
            entity.setId(doObj.getId());
        } else {
            resourceMapper.updateById(doObj);
        }
        return entity;
    }

    @Override
    public void update(ScheduleResource entity) {
        if (entity.getId() != null) {
            ScheduleResourceDO doObj = toDO(entity);
            resourceMapper.updateById(doObj);
        }
    }

    @Override
    public void deleteById(Long id) {
        resourceMapper.deleteById(id);
    }

    private ScheduleResource toEntity(ScheduleResourceDO doObj) {
        if (doObj == null) return null;
        ScheduleResource entity = new ScheduleResource();
        entity.setId(doObj.getId());
        entity.setResourceCode(doObj.getResourceCode());
        entity.setResourceName(doObj.getResourceName());
        entity.setResourceType(doObj.getResourceType() != null ? ResourceType.valueOf(doObj.getResourceType()) : null);
        entity.setStatus(doObj.getStatus() != null ? ResourceStatus.valueOf(doObj.getStatus()) : ResourceStatus.AVAILABLE);
        entity.setWorkshopId(doObj.getWorkshopId());
        entity.setCapacityPerShift(doObj.getCapacityPerShift());
        entity.setCalendarCode(doObj.getCalendarCode());
        entity.setDescription(doObj.getDescription());
        entity.setOccupiedSlots(new java.util.ArrayList<>());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    private ScheduleResourceDO toDO(ScheduleResource entity) {
        if (entity == null) return null;
        ScheduleResourceDO doObj = new ScheduleResourceDO();
        doObj.setId(entity.getId());
        doObj.setResourceCode(entity.getResourceCode());
        doObj.setResourceName(entity.getResourceName());
        doObj.setResourceType(entity.getResourceType() != null ? entity.getResourceType().name() : null);
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : ResourceStatus.AVAILABLE.name());
        doObj.setWorkshopId(entity.getWorkshopId());
        doObj.setCapacityPerShift(entity.getCapacityPerShift());
        doObj.setCalendarCode(entity.getCalendarCode());
        doObj.setDescription(entity.getDescription());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}
