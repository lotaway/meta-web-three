package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.config.StatusMachine;
import com.metawebthree.mes.domain.config.StatusConfig;
import com.metawebthree.mes.domain.config.StatusTransitionRule;
import java.util.List;
import java.util.Optional;

public interface StatusMachineRepository {
    Optional<StatusMachine> findById(Long id);
    Optional<StatusMachine> findByMachineCode(String machineCode);
    Optional<StatusMachine> findByEntityTypeAndIsDefaultTrue(String entityType);
    Optional<StatusMachine> findByEntityType(String entityType);
    StatusMachine save(StatusMachine statusMachine);
    void update(StatusMachine statusMachine);
    void deleteById(Long id);
    
    List<StatusConfig> findStatusesByMachineId(Long machineId);
    List<StatusTransitionRule> findTransitionsByMachineId(Long machineId);
    
    StatusConfig saveStatus(StatusConfig statusConfig);
    void updateStatus(StatusConfig statusConfig);
    void deleteStatusById(Long id);
    
    StatusTransitionRule saveTransition(StatusTransitionRule transition);
    void updateTransition(StatusTransitionRule transition);
    void deleteTransitionById(Long id);
}