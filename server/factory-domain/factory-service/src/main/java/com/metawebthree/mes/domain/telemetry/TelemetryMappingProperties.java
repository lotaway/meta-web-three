package com.metawebthree.mes.domain.telemetry;

import org.springframework.boot.context.properties.ConfigurationProperties;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@ConfigurationProperties(prefix = "mes.telemetry")
public class TelemetryMappingProperties {

    private boolean enabled = false;
    private String sourceName = "FIELD";
    private String timestampField = "timestamp";
    private String qualityField = "quality";
    private String unitField = "unit";
    private String deviceCodeSegment = "device";
    private String unsEnterprise = "enterprise";
    private String unsSite = "site";
    private String unsArea = "area";
    private String unsLine = "line";
    private String unsCell = "cell";
    private String unsDevice = "device";
    private String unsTag = "tag";
    private List<String> topicPatterns = new ArrayList<>();
    private Map<String, String> qualityValues = new LinkedHashMap<>();
    private Map<String, String> pointAliases = new LinkedHashMap<>();

    public TelemetryMappingProperties() {
        topicPatterns.add("mes/equipment/+/telemetry");
        qualityValues.put("good", TelemetryPoint.QUALITY_GOOD);
        qualityValues.put("ok", TelemetryPoint.QUALITY_GOOD);
        qualityValues.put("bad", TelemetryPoint.QUALITY_BAD);
        qualityValues.put("error", TelemetryPoint.QUALITY_BAD);
    }

    public boolean isEnabled() { return enabled; }
    public void setEnabled(boolean enabled) { this.enabled = enabled; }
    public String getSourceName() { return sourceName; }
    public void setSourceName(String sourceName) { this.sourceName = sourceName; }
    public String getTimestampField() { return timestampField; }
    public void setTimestampField(String timestampField) { this.timestampField = timestampField; }
    public String getQualityField() { return qualityField; }
    public void setQualityField(String qualityField) { this.qualityField = qualityField; }
    public String getUnitField() { return unitField; }
    public void setUnitField(String unitField) { this.unitField = unitField; }
    public String getDeviceCodeSegment() { return deviceCodeSegment; }
    public void setDeviceCodeSegment(String deviceCodeSegment) { this.deviceCodeSegment = deviceCodeSegment; }
    public String getUnsEnterprise() { return unsEnterprise; }
    public void setUnsEnterprise(String unsEnterprise) { this.unsEnterprise = unsEnterprise; }
    public String getUnsSite() { return unsSite; }
    public void setUnsSite(String unsSite) { this.unsSite = unsSite; }
    public String getUnsArea() { return unsArea; }
    public void setUnsArea(String unsArea) { this.unsArea = unsArea; }
    public String getUnsLine() { return unsLine; }
    public void setUnsLine(String unsLine) { this.unsLine = unsLine; }
    public String getUnsCell() { return unsCell; }
    public void setUnsCell(String unsCell) { this.unsCell = unsCell; }
    public String getUnsDevice() { return unsDevice; }
    public void setUnsDevice(String unsDevice) { this.unsDevice = unsDevice; }
    public String getUnsTag() { return unsTag; }
    public void setUnsTag(String unsTag) { this.unsTag = unsTag; }
    public List<String> getTopicPatterns() { return topicPatterns; }
    public void setTopicPatterns(List<String> topicPatterns) { this.topicPatterns = topicPatterns; }
    public Map<String, String> getQualityValues() { return qualityValues; }
    public void setQualityValues(Map<String, String> qualityValues) { this.qualityValues = qualityValues; }
    public Map<String, String> getPointAliases() { return pointAliases; }
    public void setPointAliases(Map<String, String> pointAliases) { this.pointAliases = pointAliases; }
}
