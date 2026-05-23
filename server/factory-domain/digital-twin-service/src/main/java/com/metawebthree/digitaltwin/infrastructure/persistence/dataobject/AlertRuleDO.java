package com.metawebthree.digitaltwin.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.TableName;
import com.metawebthree.common.DO.BaseDO;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Data
@NoArgsConstructor
@SuperBuilder
@EqualsAndHashCode(callSuper = false)
@TableName("alert_rules")
public class AlertRuleDO extends BaseDO {
    private Long id;
    private String ruleCode;
    private String ruleName;
    private String description;
    private String deviceType;
    private String deviceCode;
    private String workshopId;
    private String metricType;
    private String operator;
    private Double thresholdValue;
    private Integer durationSeconds;
    private String level;
    private String alertType;
    private String titleTemplate;
    private String descriptionTemplate;
    private Boolean enabled;
    private Integer cooldownSeconds;
    private Integer maxAlertsPerHour;
    private String notificationChannels;
    private String createdBy;
    private String updatedBy;
}
