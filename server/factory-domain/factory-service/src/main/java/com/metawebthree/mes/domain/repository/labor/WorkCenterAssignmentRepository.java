package com.metawebthree.mes.domain.repository.labor;

import com.metawebthree.mes.domain.entity.labor.WorkCenterAssignment;
import java.util.List;
import java.util.Optional;

public interface WorkCenterAssignmentRepository {
    Optional<WorkCenterAssignment> findById(Long id);
    List<WorkCenterAssignment> findByOperatorId(Long operatorId);
    List<WorkCenterAssignment> findByWorkCenterId(String workCenterId);
    List<WorkCenterAssignment> findByStatus(WorkCenterAssignment.AssignmentStatus status);
    List<WorkCenterAssignment> findActiveByOperatorId(Long operatorId);
    List<WorkCenterAssignment> findAll();
    WorkCenterAssignment save(WorkCenterAssignment assignment);
    void update(WorkCenterAssignment assignment);
    void deleteById(Long id);
}
