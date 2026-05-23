package com.metawebthree.production.domain.repository;

import com.metawebthree.production.domain.entity.WorkStation;
import java.util.List;
import java.util.Optional;

public interface WorkStationRepository {
    Optional<WorkStation> findById(Long id);
    Optional<WorkStation> findByStationCode(String stationCode);
    List<WorkStation> findByStatus(WorkStation.StationStatus status);
    List<WorkStation> findByWorkshopCode(String workshopCode);
    List<WorkStation> findAll();
    WorkStation save(WorkStation station);
    void delete(WorkStation station);
    List<WorkStation> findAvailableStations();
}