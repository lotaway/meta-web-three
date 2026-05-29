package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

public class SopDocument {
    private Long id;
    private String documentCode;
    private String documentName;
    private String documentType;
    private String category;
    private Integer currentVersion;
    private SopStatus status;
    private List<SopDocumentVersion> versions;
    private List<SopRouteBinding> routeBindings;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum SopStatus {
        DRAFT, ACTIVE, ARCHIVED
    }

    public void create(String documentCode, String documentName, String documentType, String category) {
        this.documentCode = documentCode;
        this.documentName = documentName;
        this.documentType = documentType;
        this.category = category;
        this.currentVersion = 0;
        this.status = SopStatus.DRAFT;
        this.versions = new ArrayList<>();
        this.routeBindings = new ArrayList<>();
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void activate() {
        this.status = SopStatus.ACTIVE;
        this.updatedAt = LocalDateTime.now();
    }

    public void archive() {
        this.status = SopStatus.ARCHIVED;
        this.updatedAt = LocalDateTime.now();
    }

    public SopDocumentVersion addVersion(String fileName, String filePath, String uploader, String changeDescription) {
        if (this.versions == null) {
            this.versions = new ArrayList<>();
        }
        for (SopDocumentVersion v : this.versions) {
            v.setIsCurrentVersion(false);
        }
        SopDocumentVersion newVersion = new SopDocumentVersion();
        newVersion.create(this.currentVersion + 1, fileName, filePath, uploader);
        newVersion.setChangeDescription(changeDescription);
        newVersion.setSopDocumentId(this.id);
        this.versions.add(newVersion);
        this.currentVersion = newVersion.getVersionNo();
        this.updatedAt = LocalDateTime.now();
        return newVersion;
    }

    public void bindRoute(String routeCode, String routeName, Integer stepNo, 
                          String processCode, String processName, 
                          Long workstationId, String workstationName) {
        if (this.routeBindings == null) {
            this.routeBindings = new ArrayList<>();
        }
        SopRouteBinding binding = new SopRouteBinding();
        binding.create(this.id, routeCode, stepNo, workstationId);
        binding.setRouteName(routeName);
        binding.setProcessCode(processCode);
        binding.setProcessName(processName);
        binding.setWorkstationName(workstationName);
        this.routeBindings.add(binding);
        this.updatedAt = LocalDateTime.now();
    }

    public void unbindRoute(String routeCode, Integer stepNo) {
        if (this.routeBindings == null) {
            return;
        }
        this.routeBindings.removeIf(b -> 
            b.getRouteCode().equals(routeCode) && 
            (stepNo == null || stepNo.equals(b.getStepNo()))
        );
        this.updatedAt = LocalDateTime.now();
    }

    public SopDocumentVersion getCurrentVersion() {
        if (this.versions == null) {
            return null;
        }
        return this.versions.stream()
            .filter(v -> Boolean.TRUE.equals(v.getIsCurrentVersion()))
            .findFirst()
            .orElse(null);
    }

    public List<SopDocumentVersion> getVersionHistory() {
        if (this.versions == null) {
            return new ArrayList<>();
        }
        return this.versions.stream()
            .sorted((v1, v2) -> v2.getVersionNo().compareTo(v1.getVersionNo()))
            .collect(java.util.stream.Collectors.toList());
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getDocumentCode() { return documentCode; }
    public void setDocumentCode(String documentCode) { this.documentCode = documentCode; }
    public String getDocumentName() { return documentName; }
    public void setDocumentName(String documentName) { this.documentName = documentName; }
    public String getDocumentType() { return documentType; }
    public void setDocumentType(String documentType) { this.documentType = documentType; }
    public String getCategory() { return category; }
    public void setCategory(String category) { this.category = category; }
    public Integer getCurrentVersionNo() { return currentVersion; }
    public void setCurrentVersion(Integer currentVersion) { this.currentVersion = currentVersion; }
    public SopStatus getStatus() { return status; }
    public void setStatus(SopStatus status) { this.status = status; }
    public List<SopDocumentVersion> getVersions() { return versions; }
    public void setVersions(List<SopDocumentVersion> versions) { this.versions = versions; }
    public List<SopRouteBinding> getRouteBindings() { return routeBindings; }
    public void setRouteBindings(List<SopRouteBinding> routeBindings) { this.routeBindings = routeBindings; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}