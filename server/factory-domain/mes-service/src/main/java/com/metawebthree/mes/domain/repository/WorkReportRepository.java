package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.WorkReport;

import java.util.List;
import java.util.Optional;

public interface WorkReportRepository {
    
    Optional<WorkReport> findById(Long id);
    
    Optional<WorkReport> findByReportNo(String reportNo);
    
    List<WorkReport> findByTaskId(Long taskId);
    
    List<WorkReport> findByWorkOrderId(Long workOrderId);
    
    List<WorkReport> findByOperatorId(String operatorId);
    
    List<WorkReport> findByStatus(WorkReport.ReportStatus status);
    
    List<WorkReport> findByWorkstationId(String workstationId);
    
    List<WorkReport> findAll();
    
    WorkReport save(WorkReport report);
    
    void update(WorkReport report);
    
    void deleteById(Long id);
}