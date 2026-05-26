package com.metawebthree.production.infrastructure.persistence.repository;

import com.metawebthree.production.domain.entity.ProductionSchedule;
import com.metawebthree.production.domain.repository.ProductionScheduleRepository;
import org.springframework.stereotype.Repository;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@Repository
public class ProductionScheduleRepositoryImpl implements ProductionScheduleRepository {
    private final Map<Long, ProductionSchedule> storage = new ConcurrentHashMap<>();
    private final Map<String, ProductionSchedule> codeIndex = new ConcurrentHashMap<>();
    private final Map<String, List<ProductionSchedule>> orderIndex = new ConcurrentHashMap<>();
    private final Map<String, List<ProductionSchedule>> stationIndex = new ConcurrentHashMap<>();

    @Override
    public Optional<ProductionSchedule> findById(Long id) {
        return Optional.ofNullable(storage.get(id));
    }

    @Override
    public Optional<ProductionSchedule> findByScheduleCode(String scheduleCode) {
        return Optional.ofNullable(codeIndex.get(scheduleCode));
    }

    @Override
    public List<ProductionSchedule> findByOrderCode(String orderCode) {
        return orderIndex.getOrDefault(orderCode, Collections.emptyList());
    }

    @Override
    public List<ProductionSchedule> findByStationCode(String stationCode) {
        return stationIndex.getOrDefault(stationCode, Collections.emptyList());
    }

    @Override
    public List<ProductionSchedule> findByStatus(ProductionSchedule.ScheduleStatus status) {
        return storage.values().stream()
            .filter(s -> s.getStatus() == status)
            .collect(Collectors.toList());
    }

    @Override
    public List<ProductionSchedule> findAll() {
        return new ArrayList<>(storage.values());
    }

    @Override
    public ProductionSchedule save(ProductionSchedule schedule) {
        if (schedule.getId() == null) {
            long maxId = storage.keySet().stream().mapToLong(Long::longValue).max().orElse(0L);
            schedule.setId(maxId + 1);
        }
        storage.put(schedule.getId(), schedule);
        
        if (schedule.getScheduleCode() != null) {
            codeIndex.put(schedule.getScheduleCode(), schedule);
        }
        
        if (schedule.getOrderCode() != null) {
            orderIndex.computeIfAbsent(schedule.getOrderCode(), k -> new ArrayList<>())
                .removeIf(s -> s.getId().equals(schedule.getId()));
            orderIndex.computeIfAbsent(schedule.getOrderCode(), k -> new ArrayList<>())
                .add(schedule);
        }
        
        if (schedule.getStationCode() != null) {
            stationIndex.computeIfAbsent(schedule.getStationCode(), k -> new ArrayList<>())
                .removeIf(s -> s.getId().equals(schedule.getId()));
            stationIndex.computeIfAbsent(schedule.getStationCode(), k -> new ArrayList<>())
                .add(schedule);
        }
        
        return schedule;
    }

    @Override
    public void delete(ProductionSchedule schedule) {
        if (schedule.getId() != null) {
            storage.remove(schedule.getId());
        }
        if (schedule.getScheduleCode() != null) {
            codeIndex.remove(schedule.getScheduleCode());
        }
        if (schedule.getOrderCode() != null) {
            orderIndex.computeIfPresent(schedule.getOrderCode(), (k, v) -> {
                v.removeIf(s -> s.getId().equals(schedule.getId()));
                return v;
            });
        }
        if (schedule.getStationCode() != null) {
            stationIndex.computeIfPresent(schedule.getStationCode(), (k, v) -> {
                v.removeIf(s -> s.getId().equals(schedule.getId()));
                return v;
            });
        }
    }
}