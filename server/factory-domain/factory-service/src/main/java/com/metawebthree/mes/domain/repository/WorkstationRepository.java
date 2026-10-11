package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.Workstation;
import java.util.List;
import java.util.Optional;

public interface WorkstationRepository {
    Optional<Workstation> findById(Long id);
    Optional<Workstation> findByWorkstationCode(String workstationCode);
    List<Workstation> findByWorkshopId(String workshopId);
    List<Workstation> findByStatus(Workstation.WorkstationStatus status);
    List<Workstation> findByType(Workstation.WorkstationType type);
    Workstation save(Workstation workstation);
    void update(Workstation workstation);
    void deleteById(Long id);
}