package com.metawebthree.digitaltwin.interfaces.dto;

import jakarta.validation.constraints.NotNull;

public class UpdateDevicePositionRequest {

    @NotNull(message = "X坐标不能为空")
    private Double x;

    @NotNull(message = "Y坐标不能为空")
    private Double y;

    private Double z;

    public Double getX() { return x; }
    public void setX(Double x) { this.x = x; }
    public Double getY() { return y; }
    public void setY(Double y) { this.y = y; }
    public Double getZ() { return z; }
    public void setZ(Double z) { this.z = z; }
}