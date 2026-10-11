package com.metawebthree.digitaltwin.interfaces.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public class CreateWorkshopRequest {

    @NotBlank(message = "车间编码不能为空")
    @Size(max = 50, message = "车间编码长度不能超过50")
    private String workshopCode;

    @NotBlank(message = "车间名称不能为空")
    @Size(max = 100, message = "车间名称长度不能超过100")
    private String workshopName;

    @Size(max = 500, message = "描述长度不能超过500")
    private String description;

    private Double area;
    private String location;
    private Double centerX;
    private Double centerY;
    private Double width;
    private Double length;

    // Getters and Setters
    public String getWorkshopCode() { return workshopCode; }
    public void setWorkshopCode(String workshopCode) { this.workshopCode = workshopCode; }
    public String getWorkshopName() { return workshopName; }
    public void setWorkshopName(String workshopName) { this.workshopName = workshopName; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public Double getArea() { return area; }
    public void setArea(Double area) { this.area = area; }
    public String getLocation() { return location; }
    public void setLocation(String location) { this.location = location; }
    public Double getCenterX() { return centerX; }
    public void setCenterX(Double centerX) { this.centerX = centerX; }
    public Double getCenterY() { return centerY; }
    public void setCenterY(Double centerY) { this.centerY = centerY; }
    public Double getWidth() { return width; }
    public void setWidth(Double width) { this.width = width; }
    public Double getLength() { return length; }
    public void setLength(Double length) { this.length = length; }
}