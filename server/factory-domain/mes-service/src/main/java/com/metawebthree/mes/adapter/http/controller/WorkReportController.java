package com.metawebthree.mes.adapter.http.controller;

import com.metawebthree.mes.domain.entity.WorkReport;
import com.metawebthree.mes.domain.repository.WorkReportRepository;
import com.metawebthree.mes.interfaces.dto.WorkReportDTO;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/mes/work-report")
@RequiredArgsConstructor
public class WorkReportController {
    
    private final WorkReportRepository workReportRepository;
    
    @PostMapping
    public ResponseEntity<WorkReportDTO> create(@RequestBody WorkReportDTO.CreateRequest request) {
        String reportNo = generateReportNo();
        
        WorkReport report = new WorkReport();
        report.create(reportNo, request.getTaskId(), request.getTaskNo(),
            request.getWorkOrderId(), request.getWorkOrderNo(),
            request.getWorkstationId(), request.getWorkstationName(),
            request.getProcessCode(), request.getProcessName(),
            request.getStepNo(), request.getOperatorId(), request.getOperatorName());
        
        if (request.getQuantity() != null) {
            report.recordOutput(
                request.getQuantity(),
                request.getQualifiedQuantity() != null ? request.getQualifiedQuantity() : 0,
                request.getDefectiveQuantity() != null ? request.getDefectiveQuantity() : 0,
                request.getDurationMinutes() != null ? request.getDurationMinutes() : 0
            );
        }
        
        if (request.getParameterValuesJson() != null) {
            report.setParameterValues(request.getParameterValuesJson());
        }
        
        report.setRemarks(request.getRemarks());
        
        WorkReport saved = workReportRepository.save(report);
        return ResponseEntity.ok(WorkReportDTO.fromEntity(saved));
    }
    
    @PutMapping("/{id}")
    public ResponseEntity<WorkReportDTO> update(
            @PathVariable Long id,
            @RequestBody WorkReportDTO.CreateRequest request) {
        
        WorkReport report = workReportRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Report not found: " + id));
        
        report.setWorkstationId(request.getWorkstationId());
        report.setWorkstationName(request.getWorkstationName());
        report.setOperatorName(request.getOperatorName());
        
        if (request.getQuantity() != null) {
            report.recordOutput(
                request.getQuantity(),
                request.getQualifiedQuantity() != null ? request.getQualifiedQuantity() : 0,
                request.getDefectiveQuantity() != null ? request.getDefectiveQuantity() : 0,
                request.getDurationMinutes() != null ? request.getDurationMinutes() : 0
            );
        }
        
        if (request.getParameterValuesJson() != null) {
            report.setParameterValues(request.getParameterValuesJson());
        }
        
        report.setRemarks(request.getRemarks());
        
        workReportRepository.update(report);
        return ResponseEntity.ok(WorkReportDTO.fromEntity(report));
    }
    
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        workReportRepository.deleteById(id);
        return ResponseEntity.ok().build();
    }
    
    @GetMapping("/{id}")
    public ResponseEntity<WorkReportDTO> getById(@PathVariable Long id) {
        return workReportRepository.findById(id)
            .map(report -> ResponseEntity.ok(WorkReportDTO.fromEntity(report)))
            .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping
    public ResponseEntity<List<WorkReportDTO>> getAll() {
        List<WorkReportDTO> reports = workReportRepository.findAll().stream()
            .map(WorkReportDTO::fromEntity)
            .collect(Collectors.toList());
        return ResponseEntity.ok(reports);
    }
    
    @GetMapping("/task/{taskId}")
    public ResponseEntity<List<WorkReportDTO>> getByTaskId(@PathVariable Long taskId) {
        List<WorkReportDTO> reports = workReportRepository.findByTaskId(taskId).stream()
            .map(WorkReportDTO::fromEntity)
            .collect(Collectors.toList());
        return ResponseEntity.ok(reports);
    }
    
    @GetMapping("/work-order/{workOrderId}")
    public ResponseEntity<List<WorkReportDTO>> getByWorkOrderId(@PathVariable Long workOrderId) {
        List<WorkReportDTO> reports = workReportRepository.findByWorkOrderId(workOrderId).stream()
            .map(WorkReportDTO::fromEntity)
            .collect(Collectors.toList());
        return ResponseEntity.ok(reports);
    }
    
    @GetMapping("/operator/{operatorId}")
    public ResponseEntity<List<WorkReportDTO>> getByOperatorId(@PathVariable String operatorId) {
        List<WorkReportDTO> reports = workReportRepository.findByOperatorId(operatorId).stream()
            .map(WorkReportDTO::fromEntity)
            .collect(Collectors.toList());
        return ResponseEntity.ok(reports);
    }
    
    @PostMapping("/{id}/submit")
    public ResponseEntity<WorkReportDTO> submit(@PathVariable Long id) {
        WorkReport report = workReportRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Report not found: " + id));
        
        report.submit();
        workReportRepository.update(report);
        
        return ResponseEntity.ok(WorkReportDTO.fromEntity(report));
    }
    
    @PostMapping("/{id}/quality-checked")
    public ResponseEntity<WorkReportDTO> qualityChecked(@PathVariable Long id) {
        WorkReport report = workReportRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Report not found: " + id));
        
        report.qualityChecked();
        workReportRepository.update(report);
        
        return ResponseEntity.ok(WorkReportDTO.fromEntity(report));
    }
    
    @PostMapping("/{id}/confirm")
    public ResponseEntity<WorkReportDTO> confirm(@PathVariable Long id) {
        WorkReport report = workReportRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Report not found: " + id));
        
        report.confirm();
        workReportRepository.update(report);
        
        return ResponseEntity.ok(WorkReportDTO.fromEntity(report));
    }
    
    @PostMapping("/{id}/cancel")
    public ResponseEntity<WorkReportDTO> cancel(@PathVariable Long id) {
        WorkReport report = workReportRepository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Report not found: " + id));
        
        report.cancel();
        workReportRepository.update(report);
        
        return ResponseEntity.ok(WorkReportDTO.fromEntity(report));
    }
    
    private String generateReportNo() {
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss"));
        String uuid = UUID.randomUUID().toString().substring(0, 8).toUpperCase();
        return "WR" + timestamp + uuid;
    }
}