package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;

import com.metawebthree.mes.domain.QcConstants;

public class DefectCode {
    private Long id;
    private String defectCode;
    private String defectName;
    private DefectCategory category;
    private DefectSeverity severity;
    private Boolean isCritical;
    private String description;
    private String dispositionGuide;
    private Boolean isEnabled;
    private Integer sortOrder;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum DefectCategory {
        DIMENSIONAL, SURFACE, MATERIAL, ASSEMBLY, ELECTRICAL, FUNCTIONAL, PACKAGING, OTHER
    }

    public enum DefectSeverity {
        CRITICAL, MAJOR, MINOR
    }

    public void create(String defectCode, String defectName, DefectCategory category, 
                       DefectSeverity severity) {
        this.defectCode = defectCode;
        this.defectName = defectName;
        this.category = category;
        this.severity = severity;
        this.isCritical = severity == DefectSeverity.CRITICAL;
        this.isEnabled = Boolean.TRUE;
        this.sortOrder = QcConstants.DEFAULT_SORT_ORDER;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void updateName(String defectName) {
        this.defectName = defectName;
        this.updatedAt = LocalDateTime.now();
    }

    public void updateCategory(DefectCategory category) {
        this.category = category;
        this.updatedAt = LocalDateTime.now();
    }

    public void updateSeverity(DefectSeverity severity) {
        this.severity = severity;
        this.isCritical = severity == DefectSeverity.CRITICAL;
        this.updatedAt = LocalDateTime.now();
    }

    public void updateDispositionGuide(String guide) {
        this.dispositionGuide = guide;
        this.updatedAt = LocalDateTime.now();
    }

    public void disable() {
        this.isEnabled = Boolean.FALSE;
        this.updatedAt = LocalDateTime.now();
    }

    public void enable() {
        this.isEnabled = Boolean.TRUE;
        this.updatedAt = LocalDateTime.now();
    }

    public void updateSortOrder(Integer order) {
        this.sortOrder = order;
        this.updatedAt = LocalDateTime.now();
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getDefectCode() { return defectCode; }
    public void setDefectCode(String defectCode) { this.defectCode = defectCode; }
    public String getDefectName() { return defectName; }
    public void setDefectName(String defectName) { this.defectName = defectName; }
    public DefectCategory getCategory() { return category; }
    public void setCategory(DefectCategory category) { this.category = category; }
    public DefectSeverity getSeverity() { return severity; }
    public void setSeverity(DefectSeverity severity) { this.severity = severity; }
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
    public LocalDateTime getUpdatedAt() { return updatedAt; }
}