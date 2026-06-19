package com.metawebthree.mes.interfaces.dto;

import java.time.LocalDateTime;

import com.metawebthree.mes.domain.entity.DefectCode;

public class DefectCodeDTO {
    
    private Long id;
    private String defectCode;
    private String defectName;
    private String category;
    private String severity;
    private Boolean isCritical;
    private String description;
    private String dispositionGuide;
    private Boolean isEnabled;
    private Integer sortOrder;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public static DefectCodeDTO fromEntity(DefectCode entity) {
        if (entity == null) return null;
        
        DefectCodeDTO dto = new DefectCodeDTO();
        dto.setId(entity.getId());
        dto.setDefectCode(entity.getDefectCode());
        dto.setDefectName(entity.getDefectName());
        dto.setCategory(entity.getCategory() != null ? entity.getCategory().name() : null);
        dto.setSeverity(entity.getSeverity() != null ? entity.getSeverity().name() : null);
        dto.setIsCritical(entity.getIsCritical());
        dto.setDescription(entity.getDescription());
        dto.setDispositionGuide(entity.getDispositionGuide());
        dto.setIsEnabled(entity.getIsEnabled());
        dto.setSortOrder(entity.getSortOrder());
        dto.setCreatedAt(entity.getCreatedAt());
        dto.setUpdatedAt(entity.getUpdatedAt());
        
        return dto;
    }
    
    public DefectCode toEntity() {
        DefectCode defectCode = new DefectCode();
        if (this.id != null) {
            defectCode.setId(this.id);
        }
        defectCode.setDefectCode(this.defectCode);
        defectCode.setDefectName(this.defectName);
        if (this.category != null) {
            defectCode.setCategory(DefectCode.DefectCategory.valueOf(this.category));
        }
        if (this.severity != null) {
            defectCode.setSeverity(DefectCode.DefectSeverity.valueOf(this.severity));
        }
        defectCode.setIsCritical(this.isCritical);
        defectCode.setDescription(this.description);
        defectCode.setDispositionGuide(this.dispositionGuide);
        defectCode.setIsEnabled(this.isEnabled);
        defectCode.setSortOrder(this.sortOrder);
        
        return defectCode;
    }
    
    // ========== Getters and Setters ==========
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getDefectCode() { return defectCode; }
    public void setDefectCode(String defectCode) { this.defectCode = defectCode; }
    public String getDefectName() { return defectName; }
    public void setDefectName(String defectName) { this.defectName = defectName; }
    public String getCategory() { return category; }
    public void setCategory(String category) { this.category = category; }
    public String getSeverity() { return severity; }
    public void setSeverity(String severity) { this.severity = severity; }
    public Boolean getIsCritical() { return isCritical; }
    public void setIsCritical(Boolean isCritical) { this.isCritical = isCritical; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public String getDispositionGuide() { return dispositionGuide; }
    public void setDispositionGuide(String dispositionGuide) { this.dispositionGuide = dispositionGuide; }
    public Boolean getIsEnabled() { return isEnabled; }
    public void setIsEnabled(Boolean isEnabled) { this.isEnabled = isEnabled; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}