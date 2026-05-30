package com.metawebthree.finance.application.command.cost.dto;

import java.math.BigDecimal;

public class CostDriverCreateCommand {
    private String driverCode;
    private String driverName;
    private String type;
    private String unit;
    private BigDecimal rate;
    private String description;

    public String getDriverCode() { return driverCode; }
    public void setDriverCode(String driverCode) { this.driverCode = driverCode; }
    public String getDriverName() { return driverName; }
    public void setDriverName(String driverName) { this.driverName = driverName; }
    public String getType() { return type; }
    public void setType(String type) { this.type = type; }
    public String getUnit() { return unit; }
    public void setUnit(String unit) { this.unit = unit; }
    public BigDecimal getRate() { return rate; }
    public void setRate(BigDecimal rate) { this.rate = rate; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
}