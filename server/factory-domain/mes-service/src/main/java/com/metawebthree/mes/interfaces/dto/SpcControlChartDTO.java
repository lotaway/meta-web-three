package com.metawebthree.mes.interfaces.dto;

import com.metawebthree.mes.domain.entity.SpcControlChart;
import com.metawebthree.mes.domain.entity.SpcControlChart.AlarmRule;
import com.metawebthree.mes.domain.entity.SpcControlChart.ChartType;
import com.metawebthree.mes.domain.entity.SpcControlChart.ControlLimits;
import com.metawebthree.mes.domain.entity.SpcControlChart.SamplingConfig;
import lombok.Data;
import java.time.LocalDateTime;
import java.util.List;

@Data
public class SpcControlChartDTO {
    private Long id;
    private String chartCode;
    private String chartName;
    private String chartType;
    private String parameterCode;
    private ControlLimitsDTO limits;
    private List<AlarmRuleDTO> alarmRules;
    private SamplingConfigDTO samplingConfig;
    private Boolean isEnabled;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    @Data
    public static class ControlLimitsDTO {
        private Double usl;
        private Double lsl;
        private Double ucl;
        private Double lcl;
        private Double centerLine;
    }
    
    @Data
    public static class AlarmRuleDTO {
        private String ruleCode;
        private String ruleName;
        private String type;
        private Integer pointCount;
        private Boolean isEnabled;
    }
    
    @Data
    public static class SamplingConfigDTO {
        private Integer sampleSize;
        private Integer frequencyMinutes;
        private Integer subgroupCount;
        private String samplingMethod;
    }
    
    public static SpcControlChartDTO fromEntity(SpcControlChart entity) {
        if (entity == null) {
            return null;
        }
        SpcControlChartDTO dto = new SpcControlChartDTO();
        dto.setId(entity.getId());
        dto.setChartCode(entity.getChartCode());
        dto.setChartName(entity.getChartName());
        dto.setChartType(entity.getChartType() != null ? entity.getChartType().name() : null);
        dto.setParameterCode(entity.getParameterCode());
        dto.setIsEnabled(entity.getIsEnabled());
        dto.setCreatedAt(entity.getCreatedAt());
        dto.setUpdatedAt(entity.getUpdatedAt());
        
        if (entity.getLimits() != null) {
            dto.setLimits(toLimitsDTO(entity.getLimits()));
        }
        
        if (entity.getAlarmRules() != null) {
            dto.setAlarmRules(entity.getAlarmRules().stream()
                .map(SpcControlChartDTO::toAlarmRuleDTO)
                .collect(java.util.stream.Collectors.toList()));
        }
        
        if (entity.getSamplingConfig() != null) {
            dto.setSamplingConfig(toSamplingConfigDTO(entity.getSamplingConfig()));
        }
        
        return dto;
    }
    
    private static ControlLimitsDTO toLimitsDTO(ControlLimits limits) {
        if (limits == null) return null;
        ControlLimitsDTO dto = new ControlLimitsDTO();
        dto.setUsl(limits.getUsl());
        dto.setLsl(limits.getLsl());
        dto.setUcl(limits.getUcl());
        dto.setLcl(limits.getLcl());
        dto.setCenterLine(limits.getCenterLine());
        return dto;
    }
    
    private static AlarmRuleDTO toAlarmRuleDTO(AlarmRule rule) {
        if (rule == null) return null;
        AlarmRuleDTO dto = new AlarmRuleDTO();
        dto.setRuleCode(rule.getRuleCode());
        dto.setRuleName(rule.getRuleName());
        dto.setType(rule.getType() != null ? rule.getType().name() : null);
        dto.setPointCount(rule.getPointCount());
        dto.setIsEnabled(rule.getIsEnabled());
        return dto;
    }
    
    private static SamplingConfigDTO toSamplingConfigDTO(SamplingConfig config) {
        if (config == null) return null;
        SamplingConfigDTO dto = new SamplingConfigDTO();
        dto.setSampleSize(config.getSampleSize());
        dto.setFrequencyMinutes(config.getFrequencyMinutes());
        dto.setSubgroupCount(config.getSubgroupCount());
        dto.setSamplingMethod(config.getSamplingMethod());
        return dto;
    }
    
    public SpcControlChart toEntity() {
        SpcControlChart entity = new SpcControlChart();
        entity.setId(this.id);
        entity.setChartCode(this.chartCode);
        entity.setChartName(this.chartName);
        if (this.chartType != null) {
            entity.setChartType(ChartType.valueOf(this.chartType));
        }
        entity.setParameterCode(this.parameterCode);
        entity.setIsEnabled(this.isEnabled);
        
        if (this.limits != null) {
            entity.setLimits(toLimitsEntity(this.limits));
        }
        
        if (this.alarmRules != null) {
            entity.setAlarmRules(this.alarmRules.stream()
                .map(SpcControlChartDTO::toAlarmRuleEntity)
                .collect(java.util.stream.Collectors.toList()));
        }
        
        if (this.samplingConfig != null) {
            entity.setSamplingConfig(toSamplingConfigEntity(this.samplingConfig));
        }
        
        return entity;
    }
    
    private static ControlLimits toLimitsEntity(ControlLimitsDTO dto) {
        if (dto == null) return null;
        ControlLimits limits = new ControlLimits();
        limits.setUsl(dto.getUsl());
        limits.setLsl(dto.getLsl());
        limits.setUcl(dto.getUcl());
        limits.setLcl(dto.getLcl());
        limits.setCenterLine(dto.getCenterLine());
        return limits;
    }
    
    private static AlarmRule toAlarmRuleEntity(AlarmRuleDTO dto) {
        if (dto == null) return null;
        AlarmRule rule = new AlarmRule();
        rule.setRuleCode(dto.getRuleCode());
        rule.setRuleName(dto.getRuleName());
        if (dto.getType() != null) {
            rule.setType(AlarmRule.AlarmType.valueOf(dto.getType()));
        }
        rule.setPointCount(dto.getPointCount());
        rule.setIsEnabled(dto.getIsEnabled());
        return rule;
    }
    
    private static SamplingConfig toSamplingConfigEntity(SamplingConfigDTO dto) {
        if (dto == null) return null;
        SamplingConfig config = new SamplingConfig();
        config.setSampleSize(dto.getSampleSize());
        config.setFrequencyMinutes(dto.getFrequencyMinutes());
        config.setSubgroupCount(dto.getSubgroupCount());
        config.setSamplingMethod(dto.getSamplingMethod());
        return config;
    }
}