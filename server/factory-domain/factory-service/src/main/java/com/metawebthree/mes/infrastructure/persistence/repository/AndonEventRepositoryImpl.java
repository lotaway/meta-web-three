package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.mes.domain.entity.AndonEvent;
import com.metawebthree.mes.domain.repository.AndonEventRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.AndonEventDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.AndonEventMapper;
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Repository;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
@org.springframework.context.annotation.Profile("!mes-in-memory")
public class AndonEventRepositoryImpl implements AndonEventRepository {
    private final AndonEventMapper mapper;
    private final ObjectMapper json;

    public AndonEventRepositoryImpl(AndonEventMapper mapper, ObjectMapper json) {
        this.mapper = mapper;
        this.json = json;
    }

    @Override
    public Optional<AndonEvent> findById(Long id) {
        QueryWrapper<AndonEventDO> query = new QueryWrapper<>();
        query.eq("id", id);
        return Optional.ofNullable(mapper.selectOne(query)).map(this::toEntity);
    }

    @Override
    public Optional<AndonEvent> findByIdForUpdate(Long id) {
        QueryWrapper<AndonEventDO> query = new QueryWrapper<>();
        query.eq("id", id).last("FOR UPDATE");
        return Optional.ofNullable(mapper.selectOne(query)).map(this::toEntity);
    }

    @Override
    public Optional<AndonEvent> findByEventNo(String eventNo) {
        QueryWrapper<AndonEventDO> query = new QueryWrapper<>();
        query.eq("event_no", eventNo);
        return Optional.ofNullable(mapper.selectOne(query)).map(this::toEntity);
    }

    @Override
    public List<AndonEvent> findByStatus(AndonEvent.AndonEventStatus status) {
        QueryWrapper<AndonEventDO> query = new QueryWrapper<>();
        query.eq("status", status);
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public List<AndonEvent> findByAndonTypeId(Long andonTypeId) {
        QueryWrapper<AndonEventDO> query = new QueryWrapper<>();
        query.eq("andon_type_id", andonTypeId);
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public List<AndonEvent> findByAndonLevelId(Long andonLevelId) {
        QueryWrapper<AndonEventDO> query = new QueryWrapper<>();
        query.eq("andon_level_id", andonLevelId);
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public List<AndonEvent> findByWorkshopId(String workshopId) {
        QueryWrapper<AndonEventDO> query = new QueryWrapper<>();
        query.eq("workshop_id", workshopId);
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public List<AndonEvent> findByEquipmentId(String equipmentId) {
        QueryWrapper<AndonEventDO> query = new QueryWrapper<>();
        query.eq("equipment_id", equipmentId);
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public List<AndonEvent> findByReporterId(String reporterId) {
        QueryWrapper<AndonEventDO> query = new QueryWrapper<>();
        query.eq("reporter_id", reporterId);
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public List<AndonEvent> findByCurrentHandlerId(String currentHandlerId) {
        QueryWrapper<AndonEventDO> query = new QueryWrapper<>();
        query.eq("current_handler_id", currentHandlerId);
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public List<AndonEvent> findByStatusAndOccurredAtBefore(AndonEvent.AndonEventStatus status, LocalDateTime time) {
        QueryWrapper<AndonEventDO> query = new QueryWrapper<>();
        query.eq("status", status);
        query.lt("occurred_at", time);
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public List<AndonEvent> findAll() {
        QueryWrapper<AndonEventDO> query = new QueryWrapper<>();
        query.orderByAsc("id");
        return mapper.selectList(query).stream().map(this::toEntity).toList();
    }

    @Override
    public AndonEvent save(AndonEvent andonEvent) {
        return persist(andonEvent);
    }

    @Override
    public void update(AndonEvent andonEvent) {
        if (andonEvent.getId() == null) throw new IllegalArgumentException("ID is required for update");
        persist(andonEvent);
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }

    private AndonEvent persist(AndonEvent entity) {
        AndonEventDO dto = new AndonEventDO();
        BeanUtils.copyProperties(entity, dto);
        dto.setUpdatedAt(LocalDateTime.now());
        if (dto.getId() == null) {
            if (dto.getCreatedAt() == null) dto.setCreatedAt(LocalDateTime.now());
            mapper.insert(dto);
        } else if (mapper.updateById(dto) != 1) {
            throw new IllegalArgumentException("AndonEvent not found: " + dto.getId());
        }
        BeanUtils.copyProperties(dto, entity);
        return entity;
    }

    private AndonEvent toEntity(AndonEventDO dto) {
        AndonEvent entity = new AndonEvent();
        BeanUtils.copyProperties(dto, entity);
        return entity;
    }
}