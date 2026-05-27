package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;

/**
 * BOM版本管理实体
 * 统一管理同一产品的多个BOM版本，支持版本历史和生效日期控制
 */
public class BomVersion {
    
    private Long id;
    private String productCode;
    private String productName;
    private List<BomVersionRecord> versions = new ArrayList<>();
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    /**
     * BOM版本记录
     */
    public static class BomVersionRecord {
        private Long id;
        private Long bomId;
        private String version;
        private String versionStatus; // DRAFT/ACTIVE/DEPRECATED
        private LocalDateTime effectiveDate;
        private LocalDateTime expiryDate;
        private String changeType; // INITIAL/UPDATE/EMERGENCY
        private String changeReason;
        private String changedBy;
        private LocalDateTime changedAt;
        
        public enum VersionStatus {
            DRAFT, ACTIVE, DEPRECATED
        }
        
        public enum ChangeType {
            INITIAL,   // 初始创建
            UPDATE,    // 常规更新
            EMERGENCY  // 紧急变更
        }
        
        public void create(Long bomId, String version, String changeType, String changedBy) {
            this.bomId = bomId;
            this.version = version;
            this.versionStatus = VersionStatus.DRAFT.name();
            this.changeType = changeType;
            this.changedBy = changedBy;
            this.changedAt = LocalDateTime.now();
        }
        
        /**
         * 激活版本
         */
        public void activate(LocalDateTime effectiveDate) {
            this.versionStatus = VersionStatus.ACTIVE.name();
            this.effectiveDate = effectiveDate;
        }
        
        /**
         * 弃用版本
         */
        public void deprecate(LocalDateTime expiryDate, String reason) {
            this.versionStatus = VersionStatus.DEPRECATED.name();
            this.expiryDate = expiryDate;
            this.changeReason = reason;
        }
        
        /**
         * 检查版本是否在生效期间内
         */
        public boolean isEffective(LocalDateTime date) {
            if (!VersionStatus.ACTIVE.name().equals(versionStatus)) {
                return false;
            }
            boolean afterEffective = effectiveDate == null || !date.isBefore(effectiveDate);
            boolean beforeExpiry = expiryDate == null || date.isBefore(expiryDate);
            return afterEffective && beforeExpiry;
        }
        
        // Getters and Setters
        public Long getId() { return id; }
        public void setId(Long id) { this.id = id; }
        public Long getBomId() { return bomId; }
        public void setBomId(Long bomId) { this.bomId = bomId; }
        public String getVersion() { return version; }
        public void setVersion(String version) { this.version = version; }
        public String getVersionStatus() { return versionStatus; }
        public void setVersionStatus(String versionStatus) { this.versionStatus = versionStatus; }
        public LocalDateTime getEffectiveDate() { return effectiveDate; }
        public void setEffectiveDate(LocalDateTime effectiveDate) { this.effectiveDate = effectiveDate; }
        public LocalDateTime getExpiryDate() { return expiryDate; }
        public void setExpiryDate(LocalDateTime expiryDate) { this.expiryDate = expiryDate; }
        public String getChangeType() { return changeType; }
        public void setChangeType(String changeType) { this.changeType = changeType; }
        public String getChangeReason() { return changeReason; }
        public void setChangeReason(String changeReason) { this.changeReason = changeReason; }
        public String getChangedBy() { return changedBy; }
        public void setChangedBy(String changedBy) { this.changedBy = changedBy; }
        public LocalDateTime getChangedAt() { return changedAt; }
        public void setChangedAt(LocalDateTime changedAt) { this.changedAt = changedAt; }
    }
    
    /**
     * 初始化版本管理
     */
    public void initialize(String productCode, String productName) {
        this.productCode = productCode;
        this.productName = productName;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
    
    /**
     * 添加版本记录
     */
    public void addVersion(BomVersionRecord record) {
        // 同一产品同一版本号只能有一个记录
        boolean exists = versions.stream()
                .anyMatch(v -> v.getVersion().equals(record.getVersion()));
        if (exists) {
            throw new IllegalArgumentException("Version already exists: " + record.getVersion());
        }
        this.versions.add(record);
        this.updatedAt = LocalDateTime.now();
    }
    
    /**
     * 根据版本号查找版本
     */
    public Optional<BomVersionRecord> findByVersion(String version) {
        return this.versions.stream()
                .filter(v -> v.getVersion().equals(version))
                .findFirst();
    }
    
    /**
     * 获取当前生效版本（按生效日期判断）
     */
    public Optional<BomVersionRecord> getEffectiveVersion(LocalDateTime date) {
        return this.versions.stream()
                .filter(v -> v.isEffective(date))
                .max(Comparator.comparing(BomVersionRecord::getEffectiveDate));
    }
    
    /**
     * 获取最新版本
     */
    public Optional<BomVersionRecord> getLatestVersion() {
        return this.versions.stream()
                .max(Comparator.comparing(BomVersionRecord::getChangedAt));
    }
    
    /**
     * 激活指定版本
     */
    public void activateVersion(String version, LocalDateTime effectiveDate) {
        BomVersionRecord target = findByVersion(version)
                .orElseThrow(() -> new IllegalArgumentException("Version not found: " + version));
        
        // 先停用其他版本
        this.versions.stream()
                .filter(v -> BomVersionRecord.VersionStatus.ACTIVE.name().equals(v.getVersionStatus()))
                .forEach(v -> v.deprecate(effectiveDate, "Superseded by version " + version));
        
        // 激活目标版本
        target.activate(effectiveDate);
        this.updatedAt = LocalDateTime.now();
    }
    
    /**
     * 获取版本历史（按时间倒序）
     */
    public List<BomVersionRecord> getVersionHistory() {
        return this.versions.stream()
                .sorted(Comparator.comparing(BomVersionRecord::getChangedAt).reversed())
                .toList();
    }
    
    /**
     * 根据日期自动选择BOM版本
     */
    public Optional<Long> selectBomIdByDate(LocalDateTime date) {
        return getEffectiveVersion(date)
                .map(BomVersionRecord::getBomId);
    }
    
    /**
     * 获取版本数量
     */
    public int getVersionCount() {
        return this.versions.size();
    }
    
    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getProductCode() { return productCode; }
    public void setProductCode(String productCode) { this.productCode = productCode; }
    public String getProductName() { return productName; }
    public void setProductName(String productName) { this.productName = productName; }
    public List<BomVersionRecord> getVersions() { return versions; }
    public void setVersions(List<BomVersionRecord> versions) { this.versions = versions; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}