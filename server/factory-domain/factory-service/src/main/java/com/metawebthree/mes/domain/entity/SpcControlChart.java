package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class SpcControlChart {
    private Long id;
    private String chartCode;
    private String chartName;
    private ChartType chartType;
    private String parameterCode;
    private ControlLimits limits;
    private List<AlarmRule> alarmRules;
    private SamplingConfig samplingConfig;
    private Boolean isEnabled;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public enum ChartType {
        XBAR_R, XBAR_S, X_MR, P_CHART, NP_CHART, C_CHART, CU_CHART
    }

    public static class ControlLimits {
        private Double usl;
        private Double lsl;
        private Double ucl;
        private Double lcl;
        private Double centerLine;

        public Double getUsl() { return usl; }
        public void setUsl(Double usl) { this.usl = usl; }
        public Double getLsl() { return lsl; }
        public void setLsl(Double lsl) { this.lsl = lsl; }
        public Double getUcl() { return ucl; }
        public void setUcl(Double ucl) { this.ucl = ucl; }
        public Double getLcl() { return lcl; }
        public void setLcl(Double lcl) { this.lcl = lcl; }
        public Double getCenterLine() { return centerLine; }
        public void setCenterLine(Double centerLine) { this.centerLine = centerLine; }
    }

    public static class AlarmRule {
        private String ruleCode;
        private String ruleName;
        private AlarmType type;
        private Integer pointCount;
        private Boolean isEnabled;

        public enum AlarmType {
            SINGLE_POINT_OUTSIDE, SEVEN_POINTS_ONE_SIDE, SIX_POINTS_TREND, 
            FOURTEEN_POINTS_ALTERNATING, THREE_OF_FOUR_OUTSIDE_2SIGMA,
            FIVE_OF_FIVE_OUTSIDE_1SIGMA, EIGHT_OF_EIGHT_ON_ONE_SIDE
        }

        public String getRuleCode() { return ruleCode; }
        public void setRuleCode(String ruleCode) { this.ruleCode = ruleCode; }
        public String getRuleName() { return ruleName; }
        public void setRuleName(String ruleName) { this.ruleName = ruleName; }
        public AlarmType getType() { return type; }
        public void setType(AlarmType type) { this.type = type; }
        public Integer getPointCount() { return pointCount; }
        public void setPointCount(Integer pointCount) { this.pointCount = pointCount; }
        public Boolean getIsEnabled() { return isEnabled; }
        public void setIsEnabled(Boolean isEnabled) { this.isEnabled = isEnabled; }
    }

    public static class SamplingConfig {
        private Integer sampleSize;
        private Integer frequencyMinutes;
        private Integer subgroupCount;
        private String samplingMethod;

        public Integer getSampleSize() { return sampleSize; }
        public void setSampleSize(Integer sampleSize) { this.sampleSize = sampleSize; }
        public Integer getFrequencyMinutes() { return frequencyMinutes; }
        public void setFrequencyMinutes(Integer frequencyMinutes) { this.frequencyMinutes = frequencyMinutes; }
        public Integer getSubgroupCount() { return subgroupCount; }
        public void setSubgroupCount(Integer subgroupCount) { this.subgroupCount = subgroupCount; }
        public String getSamplingMethod() { return samplingMethod; }
        public void setSamplingMethod(String samplingMethod) { this.samplingMethod = samplingMethod; }
    }

    public void create(String chartCode, String chartName, ChartType chartType, 
                       String parameterCode) {
        this.chartCode = chartCode;
        this.chartName = chartName;
        this.chartType = chartType;
        this.parameterCode = parameterCode;
        this.limits = new ControlLimits();
        this.alarmRules = new ArrayList<>();
        this.samplingConfig = new SamplingConfig();
        this.isEnabled = Boolean.TRUE;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    public void setControlLimits(Double usl, Double lsl, Double ucl, Double lcl, Double cl) {
        this.limits = new ControlLimits();
        this.limits.setUsl(usl);
        this.limits.setLsl(lsl);
        this.limits.setUcl(ucl);
        this.limits.setLcl(lcl);
        this.limits.setCenterLine(cl);
        this.updatedAt = LocalDateTime.now();
    }

    public void addAlarmRule(AlarmRule.AlarmType type, String ruleName, Integer pointCount) {
        if (this.alarmRules == null) {
            this.alarmRules = new ArrayList<>();
        }
        AlarmRule rule = new AlarmRule();
        rule.setRuleCode(type.name());
        rule.setRuleName(ruleName);
        rule.setType(type);
        rule.setPointCount(pointCount);
        rule.setIsEnabled(Boolean.TRUE);
        this.alarmRules.add(rule);
        this.updatedAt = LocalDateTime.now();
    }

    public void addDefaultAlarmRules() {
        this.alarmRules = new ArrayList<>();
        addAlarmRule(AlarmRule.AlarmType.SINGLE_POINT_OUTSIDE, "单点超出控制限", 1);
        addAlarmRule(AlarmRule.AlarmType.SEVEN_POINTS_ONE_SIDE, "连续7点在中心线一侧", 7);
        addAlarmRule(AlarmRule.AlarmType.SIX_POINTS_TREND, "连续6点趋势", 6);
        addAlarmRule(AlarmRule.AlarmType.FOURTEEN_POINTS_ALTERNATING, "连续14点交替上下", 14);
        addAlarmRule(AlarmRule.AlarmType.THREE_OF_FOUR_OUTSIDE_2SIGMA, "3/4点超出2σ", 4);
    }

    public void removeAlarmRule(String ruleCode) {
        if (alarmRules != null) {
            alarmRules.removeIf(r -> ruleCode.equals(r.getRuleCode()));
            this.updatedAt = LocalDateTime.now();
        }
    }

    public void setSamplingConfig(Integer sampleSize, Integer frequencyMinutes, 
                                  Integer subgroupCount, String method) {
        this.samplingConfig = new SamplingConfig();
        this.samplingConfig.setSampleSize(sampleSize);
        this.samplingConfig.setFrequencyMinutes(frequencyMinutes);
        this.samplingConfig.setSubgroupCount(subgroupCount);
        this.samplingConfig.setSamplingMethod(method);
        this.updatedAt = LocalDateTime.now();
    }

    public Optional<AlarmRule> findAlarmRule(AlarmRule.AlarmType type) {
        if (alarmRules == null) {
            return Optional.empty();
        }
        return alarmRules.stream()
            .filter(r -> type.equals(r.getType()))
            .findFirst();
    }

    public Boolean checkAlarm(List<Double> values) {
        if (values == null || values.isEmpty() || alarmRules == null) {
            return Boolean.FALSE;
        }
        for (AlarmRule rule : alarmRules) {
            if (!Boolean.TRUE.equals(rule.getIsEnabled())) {
                continue;
            }
            if (evaluateRule(rule, values)) {
                return Boolean.TRUE;
            }
        }
        return Boolean.FALSE;
    }

    private Boolean evaluateRule(AlarmRule rule, List<Double> values) {
        if (values.size() < rule.getPointCount()) {
            return Boolean.FALSE;
        }
        return switch (rule.getType()) {
            case SINGLE_POINT_OUTSIDE -> checkSinglePointOut(values);
            case SEVEN_POINTS_ONE_SIDE -> checkSevenOneSide(values);
            case SIX_POINTS_TREND -> checkSixTrend(values);
            default -> Boolean.FALSE;
        };
    }

    private Boolean checkSinglePointOut(List<Double> values) {
        Double latest = values.get(values.size() - 1);
        if (limits == null) {
            return Boolean.FALSE;
        }
        return (limits.getUcl() != null && latest > limits.getUcl()) ||
               (limits.getLcl() != null && latest < limits.getLcl());
    }

    private Boolean checkSevenOneSide(List<Double> values) {
        if (limits == null || limits.getCenterLine() == null) {
            return Boolean.FALSE;
        }
        Double cl = limits.getCenterLine();
        int count = 0;
        for (int i = values.size() - 7; i < values.size(); i++) {
            if (values.get(i) > cl) {
                count++;
            }
        }
        return count >= 7;
    }

    private Boolean checkSixTrend(List<Double> values) {
        if (values.size() < 6) {
            return Boolean.FALSE;
        }
        boolean allIncreasing = true;
        boolean allDecreasing = true;
        for (int i = values.size() - 6; i < values.size() - 1; i++) {
            if (values.get(i) >= values.get(i + 1)) {
                allIncreasing = false;
            }
            if (values.get(i) <= values.get(i + 1)) {
                allDecreasing = false;
            }
        }
        return allIncreasing || allDecreasing;
    }

    public void disable() {
        this.isEnabled = Boolean.FALSE;
        this.updatedAt = LocalDateTime.now();
    }

    public void enable() {
        this.isEnabled = Boolean.TRUE;
        this.updatedAt = LocalDateTime.now();
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getChartCode() { return chartCode; }
    public void setChartCode(String chartCode) { this.chartCode = chartCode; }
    public String getChartName() { return chartName; }
    public void setChartName(String chartName) { this.chartName = chartName; }
    public ChartType getChartType() { return chartType; }
    public void setChartType(ChartType chartType) { this.chartType = chartType; }
    public String getParameterCode() { return parameterCode; }
    public void setParameterCode(String parameterCode) { this.parameterCode = parameterCode; }
    public ControlLimits getLimits() { return limits; }
    public void setLimits(ControlLimits limits) { this.limits = limits; }
    public List<AlarmRule> getAlarmRules() { return alarmRules; }
    public void setAlarmRules(List<AlarmRule> alarmRules) { this.alarmRules = alarmRules; }
    public SamplingConfig getSamplingConfig() { return samplingConfig; }
    public void setSamplingConfig(SamplingConfig samplingConfig) { this.samplingConfig = samplingConfig; }
    public Boolean getIsEnabled() { return isEnabled; }
    public void setIsEnabled(Boolean isEnabled) { this.isEnabled = isEnabled; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
}