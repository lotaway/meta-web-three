package com.metawebthree.mes.domain.entity.labor;

import java.time.LocalDateTime;

public class OperatorSkill {

    public enum SkillLevel {
        TRAINEE, JUNIOR, MIDDLE, SENIOR, MASTER
    }

    private Long id;
    private Long operatorId;
    private String skillCode;
    private String skillName;
    private SkillLevel skillLevel;
    private boolean certified;
    private LocalDateTime certifiedAt;
    private LocalDateTime expiryAt;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public void create(Long operatorId, String skillCode, String skillName, SkillLevel level) {
        this.operatorId = operatorId;
        this.skillCode = skillCode;
        this.skillName = skillName;
        this.skillLevel = level;
        this.certified = false;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void certify(LocalDateTime expiryAt) {
        this.certified = true;
        this.certifiedAt = LocalDateTime.now();
        this.expiryAt = expiryAt;
        this.updatedAt = LocalDateTime.now();
    }

    public void revokeCertification() {
        this.certified = false;
        this.certifiedAt = null;
        this.expiryAt = null;
        this.updatedAt = LocalDateTime.now();
    }

    public void upgrade(SkillLevel newLevel) {
        if (newLevel.ordinal() <= this.skillLevel.ordinal()) {
            throw new IllegalArgumentException("New level must be higher than current level");
        }
        this.skillLevel = newLevel;
        this.updatedAt = LocalDateTime.now();
    }

    public boolean isExpired() {
        return certified && expiryAt != null && LocalDateTime.now().isAfter(expiryAt);
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getOperatorId() { return operatorId; }
    public void setOperatorId(Long operatorId) { this.operatorId = operatorId; }
    public String getSkillCode() { return skillCode; }
    public void setSkillCode(String skillCode) { this.skillCode = skillCode; }
    public String getSkillName() { return skillName; }
    public void setSkillName(String skillName) { this.skillName = skillName; }
    public SkillLevel getSkillLevel() { return skillLevel; }
    public void setSkillLevel(SkillLevel skillLevel) { this.skillLevel = skillLevel; }
    public boolean isCertified() { return certified; }
    public void setCertified(boolean certified) { this.certified = certified; }
    public LocalDateTime getCertifiedAt() { return certifiedAt; }
    public void setCertifiedAt(LocalDateTime certifiedAt) { this.certifiedAt = certifiedAt; }
    public LocalDateTime getExpiryAt() { return expiryAt; }
    public void setExpiryAt(LocalDateTime expiryAt) { this.expiryAt = expiryAt; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}
