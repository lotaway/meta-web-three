package com.metawebthree.mes.domain.entity.labor;

import java.time.LocalDateTime;
import java.util.List;

public class Operator {

    public enum OperatorStatus {
        ACTIVE, INACTIVE, ON_LEAVE, TERMINATED
    }

    private Long id;
    private String operatorCode;
    private String operatorName;
    private String department;
    private String jobTitle;
    private String shiftGroup;
    private OperatorStatus status;
    private String phone;
    private String email;
    private String idCardNo;
    private LocalDateTime hireDate;
    private List<OperatorSkill> skills;
    private String remark;
    private String createdBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public void create(String operatorCode, String operatorName, String department, String shiftGroup) {
        this.operatorCode = operatorCode;
        this.operatorName = operatorName;
        this.department = department;
        this.shiftGroup = shiftGroup;
        this.status = OperatorStatus.ACTIVE;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void activate() {
        if (this.status == OperatorStatus.TERMINATED) {
            throw new IllegalStateException("Cannot activate a terminated operator");
        }
        this.status = OperatorStatus.ACTIVE;
        this.updatedAt = LocalDateTime.now();
    }

    public void inactivate() {
        this.status = OperatorStatus.INACTIVE;
        this.updatedAt = LocalDateTime.now();
    }

    public void setOnLeave() {
        this.status = OperatorStatus.ON_LEAVE;
        this.updatedAt = LocalDateTime.now();
    }

    public void terminate() {
        this.status = OperatorStatus.TERMINATED;
        this.updatedAt = LocalDateTime.now();
    }

    public void addSkill(OperatorSkill skill) {
        if (this.skills == null) {
            this.skills = new java.util.ArrayList<>();
        }
        this.skills.add(skill);
        this.updatedAt = LocalDateTime.now();
    }

    public void removeSkill(String skillCode) {
        if (this.skills != null) {
            this.skills.removeIf(s -> s.getSkillCode().equals(skillCode));
            this.updatedAt = LocalDateTime.now();
        }
    }

    public boolean hasSkill(String skillCode) {
        if (this.skills == null) return false;
        return this.skills.stream().anyMatch(s -> s.getSkillCode().equals(skillCode));
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getOperatorCode() { return operatorCode; }
    public void setOperatorCode(String operatorCode) { this.operatorCode = operatorCode; }
    public String getOperatorName() { return operatorName; }
    public void setOperatorName(String operatorName) { this.operatorName = operatorName; }
    public String getDepartment() { return department; }
    public void setDepartment(String department) { this.department = department; }
    public String getJobTitle() { return jobTitle; }
    public void setJobTitle(String jobTitle) { this.jobTitle = jobTitle; }
    public String getShiftGroup() { return shiftGroup; }
    public void setShiftGroup(String shiftGroup) { this.shiftGroup = shiftGroup; }
    public OperatorStatus getStatus() { return status; }
    public void setStatus(OperatorStatus status) { this.status = status; }
    public String getPhone() { return phone; }
    public void setPhone(String phone) { this.phone = phone; }
    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }
    public String getIdCardNo() { return idCardNo; }
    public void setIdCardNo(String idCardNo) { this.idCardNo = idCardNo; }
    public LocalDateTime getHireDate() { return hireDate; }
    public void setHireDate(LocalDateTime hireDate) { this.hireDate = hireDate; }
    public List<OperatorSkill> getSkills() { return skills; }
    public void setSkills(List<OperatorSkill> skills) { this.skills = skills; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
    public String getCreatedBy() { return createdBy; }
    public void setCreatedBy(String createdBy) { this.createdBy = createdBy; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}
