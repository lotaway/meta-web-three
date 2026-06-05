package com.metawebthree.mes.domain.repository.scheduling;

import com.metawebthree.mes.domain.entity.scheduling.ScheduleResource;

import java.util.List;
import java.util.Optional;

public interface ScheduleResourceRepository {
    Optional<ScheduleResource> findById(Long id);
    Optional<ScheduleResource> findByResourceCode(String resourceCode);
    List<ScheduleResource> findByWorkshopId(String workshopId);
    List<ScheduleResource> findByResourceType(ScheduleResource.ResourceType resourceType);
    List<ScheduleResource> findAll();
    ScheduleResource save(ScheduleResource scheduleResource);
    void update(ScheduleResource scheduleResource);
    void deleteById(Long id);
}
