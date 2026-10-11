package com.metawebthree.digitaltwin.interfaces.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public class CreateAlertRequest {

    @NotBlank(message = "告警编码不能为空")
    @Size(max = 50, message = "告警编码长度不能超过50")
    private String alertCode;

    @NotBlank(message = "设备编码不能为空")
    private String deviceCode;

    private String workshopId;

    @NotBlank(message = "告警级别不能为空")
    private String level;

    @NotBlank(message = "告警类型不能为空")
    private String type;

    @NotBlank(message = "告警标题不能为空")
    @Size(max = 200, message = "告警标题长度不能超过200")
    private String title;

    @Size(max = 1000, message = "描述长度不能超过1000")
    private String description;

    // Getters and Setters
    public String getAlertCode() { return alertCode; }
    public void setAlertCode(String alertCode) { this.alertCode = alertCode; }
    public String getDeviceCode() { return deviceCode; }
    public void setDeviceCode(String deviceCode) { this.deviceCode = deviceCode; }
    public String getWorkshopId() { return workshopId; }
    public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
    public String getLevel() { return level; }
    public void setLevel(String level) { this.level = level; }
    public String getType() { return type; }
    public void setType(String type) { this.type = type; }
    public String getTitle() { return title; }
    public void setTitle(String title) { this.title = title; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
}