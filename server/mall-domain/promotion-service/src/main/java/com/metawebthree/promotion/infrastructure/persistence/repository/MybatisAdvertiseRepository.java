package com.metawebthree.promotion.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.promotion.domain.model.Advertise;
import com.metawebthree.promotion.domain.ports.AdvertiseRepository;
import com.metawebthree.promotion.infrastructure.persistence.mapper.AdvertiseMapper;
import com.metawebthree.promotion.infrastructure.persistence.model.AdvertiseRecord;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.stream.Collectors;

@Repository
public class MybatisAdvertiseRepository implements AdvertiseRepository {
    private final AdvertiseMapper mapper;

    public MybatisAdvertiseRepository(AdvertiseMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public void save(Advertise advertise) {
        mapper.insert(toRecord(advertise));
    }

    @Override
    public void update(Advertise advertise) {
        mapper.updateById(toRecord(advertise));
    }

    @Override
    public void delete(Long id) {
        mapper.deleteById(id);
    }

    @Override
    public Advertise findById(Long id) {
        AdvertiseRecord record = mapper.selectById(id);
        return record == null ? null : toDomain(record);
    }

    @Override
    public List<Advertise> list(String name, Integer type, String endTime, Integer status) {
        LambdaQueryWrapper<AdvertiseRecord> query = new LambdaQueryWrapper<>();
        if (name != null) query.like(AdvertiseRecord::getName, name);
        if (type != null) query.eq(AdvertiseRecord::getType, type);
        if (status != null) query.eq(AdvertiseRecord::getStatus, status);
        return mapper.selectList(query).stream().map(this::toDomain).collect(Collectors.toList());
    }

    @Override
    public List<Advertise> listAvailable(Integer type) {
        return mapper.selectList(new LambdaQueryWrapper<AdvertiseRecord>()
                .eq(AdvertiseRecord::getType, type)
                .eq(AdvertiseRecord::getStatus, 1)
                .orderByDesc(AdvertiseRecord::getSort))
                .stream().map(this::toDomain).collect(Collectors.toList());
    }

    private AdvertiseRecord toRecord(Advertise domain) {
        AdvertiseRecord record = new AdvertiseRecord();
        record.setId(domain.getId());
        record.setName(domain.getName());
        record.setType(domain.getType());
        record.setPic(domain.getPic());
        record.setStartTime(domain.getStartTime());
        record.setEndTime(domain.getEndTime());
        record.setStatus(domain.getStatus());
        record.setClickCount(domain.getClickCount());
        record.setOrderCount(domain.getOrderCount());
        record.setUrl(domain.getUrl());
        record.setNote(domain.getNote());
        record.setSort(domain.getSort());
        return record;
    }

    private Advertise toDomain(AdvertiseRecord record) {
        return Advertise.builder()
                .id(record.getId())
                .name(record.getName())
                .type(record.getType())
                .pic(record.getPic())
                .startTime(record.getStartTime())
                .endTime(record.getEndTime())
                .status(record.getStatus())
                .clickCount(record.getClickCount())
                .orderCount(record.getOrderCount())
                .url(record.getUrl())
                .note(record.getNote())
                .sort(record.getSort())
                .build();
    }
}
