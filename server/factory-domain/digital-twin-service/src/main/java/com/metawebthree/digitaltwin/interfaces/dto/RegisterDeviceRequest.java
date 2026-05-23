package com.metawebthree.digitaltwin.interfaces.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public class RegisterDeviceRequest {

    @NotBlank(message = "设备编码不能为空")
    @Size(max = 50, message = "设备编码长度不能超过50")
    private String deviceCode;

    @NotBlank(message = "设备名称不能为空")
    @Size(max = 100, message = "设备名称长度不能超过100")
    private String deviceName;

    @NotBlank(message = "设备类型不能为空")
    @Size(max = 20, message = "设备类型长度不能超过20")
    private String deviceType;

    @Size(max = 50, message = "车间ID长度不能超过50")
    private String workshopId;

    @Size(max = 50, message = "产线ID长度不能超过50")
    private String productionLineId;

    // Getters and Setters
    public String getDeviceCode() { return deviceCode; }
    public void setDeviceCode(String deviceCode) { this.deviceCode = deviceCode; }
    public String getDeviceName() { return deviceName; }
    public void setDeviceName(String deviceName) { this.deviceName = deviceName; }
    public String getDeviceType() { return deviceType; }
    public void setDeviceType(String deviceType) { this.deviceType = deviceType; }
    public String getWorkshopId() { return workshopId; }
    public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
    public String getProductionLineId() { return productionLineId; }
    public void setProductionLineId(String productionLineId) { this.productionLineId = productionLineId; }
}