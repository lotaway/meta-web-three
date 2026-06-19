package com.metawebthree.finance.application.command.asset.dto;


public class DepreciationGenerateCommand {
    private String depreciationPeriod;
    private String depreciationMethod;
    private Long departmentId;

    public String getDepreciationPeriod() { return depreciationPeriod; }
    public String getDepreciationMethod() { return depreciationMethod; }
    public Long getDepartmentId() { return departmentId; }

    public void setDepreciationPeriod(String depreciationPeriod) { this.depreciationPeriod = depreciationPeriod; }
    public void setDepreciationMethod(String depreciationMethod) { this.depreciationMethod = depreciationMethod; }
    public void setDepartmentId(Long departmentId) { this.departmentId = departmentId; }
}