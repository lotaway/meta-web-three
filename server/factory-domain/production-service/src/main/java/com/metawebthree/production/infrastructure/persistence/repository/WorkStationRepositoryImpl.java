package com.metawebthree.production.infrastructure.persistence.repository;

import com.metawebthree.production.domain.entity.WorkStation;
import com.metawebthree.production.domain.repository.WorkStationRepository;
import org.springframework.stereotype.Repository;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@Repository
public class WorkStationRepositoryImpl implements WorkStationRepository {
    private final Map<Long, WorkStation> storage = new ConcurrentHashMap<>();
    private final Map<String, WorkStation> codeIndex = new ConcurrentHashMap<>();
    private Long idGenerator = 1L;

    @Override
    public Optional<WorkStation> findById(Long id) {
        return Optional.ofNullable(storage.get(id));
    }

    @Override
    public Optional<WorkStation> findByStationCode(String stationCode) {
        return Optional.ofNullable(codeIndex.get(stationCode));
    }

    @Override
    public List<WorkStation> findByStatus(WorkStation.StationStatus status) {
        return storage.values().stream()
            .filter(s -> s.getStatus() == status)
            .collect(Collectors.toList());
    }

    @Override
    public List<WorkStation> findByWorkshopCode(String workshopCode) {
        return storage.values().stream()
            .filter(s -> workshopCode.equals(s.getWorkshopCode()))
            .collect(Collectors.toList());
    }

    @Override
    public List<WorkStation> findAll() {
        return new ArrayList<>(storage.values());
    }

    @Override
    public WorkStation save(WorkStation station) {
        if (station.getId() == null) {
            station.setId(idGenerator++);
        }
        storage.put(station.getId(), station);
        if (station.getStationCode() != null) {
            codeIndex.put(station.getStationCode(), station);
        }
        return station;
    }

    @Override
    public void delete(WorkStation station) {
        if (station.getId() != null) {
            storage.remove(station.getId());
        }
        if (station.getStationCode() != null) {
            codeIndex.remove(station.getStationCode());
        }
    }

    @Override
    public List<WorkStation> findAvailableStations() {
        return storage.values().stream()
            .filter(s -> s.getStatus() == WorkStation.StationStatus.IDLE)
            .collect(Collectors.toList());
    }
}