package com.metawebthree.mes.domain.entity.scada;

public class TelemetryMetric {
    private String metricCode;
    private String metricName;
    private Double value;
    private String unit;
    private String quality;
    private Double upperLimit;
    private Double lowerLimit;

    public String getMetricCode() { return metricCode; }
    public void setMetricCode(String metricCode) { this.metricCode = metricCode; }
    public String getMetricName() { return metricName; }
    public void setMetricName(String metricName) { this.metricName = metricName; }
    public Double getValue() { return value; }
    public void setValue(Double value) { this.value = value; }
    public String getUnit() { return unit; }
    public void setUnit(String unit) { this.unit = unit; }
    public String getQuality() { return quality; }
    public void setQuality(String quality) { this.quality = quality; }
    public Double getUpperLimit() { return upperLimit; }
    public void setUpperLimit(Double upperLimit) { this.upperLimit = upperLimit; }
    public Double getLowerLimit() { return lowerLimit; }
    public void setLowerLimit(Double lowerLimit) { this.lowerLimit = lowerLimit; }
}
