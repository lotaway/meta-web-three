package com.metawebthree.mes.interfaces.dto;

import com.metawebthree.mes.domain.entity.NonConformanceDisposition;
import com.metawebthree.mes.domain.entity.NonConformanceDisposition.DispositionStep;
import com.metawebthree.mes.domain.entity.NonConformanceDisposition.DispositionType;
import lombok.Data;
import java.time.LocalDateTime;
import java.util.List;

@Data
public class NonConformanceDispositionDTO {
    private Long id;
    private String dispositionCode;
    private String dispositionName;
    private String type;
    private List<DispositionStepDTO> steps;
    private Boolean isEnabled;
    private Integer sortOrder;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    @Data
    public static class DispositionStepDTO {
        private Integer stepOrder;
        private String stepName;
        private String action;
        private String assigneeRole;
        private Boolean requiresApproval;
        private Integer timeoutHours;
    }
    
    public static NonConformanceDispositionDTO fromEntity(NonConformanceDisposition entity) {
        if (entity == null) {
            return null;
        }
        NonConformanceDispositionDTO dto = new NonConformanceDispositionDTO();
        dto.setId(entity.getId());
        dto.setDispositionCode(entity.getDispositionCode());
        dto.setDispositionName(entity.getDispositionName());
        dto.setType(entity.getType() != null ? entity.getType().name() : null);
        dto.setIsEnabled(entity.getIsEnabled());
        dto.setSortOrder(entity.getSortOrder());
        dto.setCreatedAt(entity.getCreatedAt());
        dto.setUpdatedAt(entity.getUpdatedAt());
        
        if (entity.getSteps() != null) {
            dto.setSteps(entity.getSteps().stream()
                .map(NonConformanceDispositionDTO::toStepDTO)
                .collect(java.util.stream.Collectors.toList()));
        }
        
        return dto;
    }
    
    private static DispositionStepDTO toStepDTO(DispositionStep step) {
        if (step == null) {
            return null;
        }
        DispositionStepDTO dto = new DispositionStepDTO();
        dto.setStepOrder(step.getStepOrder());
        dto.setStepName(step.getStepName());
        dto.setAction(step.getAction() != null ? step.getAction().name() : null);
        dto.setAssigneeRole(step.getAssigneeRole());
        dto.setRequiresApproval(step.getRequiresApproval());
        dto.setTimeoutHours(step.getTimeoutHours());
        return dto;
    }
    
    public NonConformanceDisposition toEntity() {
        NonConformanceDisposition entity = new NonConformanceDisposition();
        entity.setId(this.id);
        entity.setDispositionCode(this.dispositionCode);
        entity.setDispositionName(this.dispositionName);
        if (this.type != null) {
            entity.setType(DispositionType.valueOf(this.type));
        }
        entity.setIsEnabled(this.isEnabled);
        entity.setSortOrder(this.sortOrder);
        
        if (this.steps != null) {
            entity.setSteps(this.steps.stream()
                .map(NonConformanceDispositionDTO::toStepEntity)
                .collect(java.util.stream.Collectors.toList()));
        }
        
        return entity;
    }
    
    private static DispositionStep toStepEntity(DispositionStepDTO dto) {
        if (dto == null) {
            return null;
        }
        DispositionStep step = new DispositionStep();
        step.setStepOrder(dto.getStepOrder());
        step.setStepName(dto.getStepName());
        if (dto.getAction() != null) {
            step.setAction(DispositionStep.StepAction.valueOf(dto.getAction()));
        }
        step.setAssigneeRole(dto.getAssigneeRole());
        step.setRequiresApproval(dto.getRequiresApproval());
        step.setTimeoutHours(dto.getTimeoutHours());
        return step;
    }
}