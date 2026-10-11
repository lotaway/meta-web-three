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
@TableName("alerts")
public class AlertDO extends BaseDO {
    private Long id;
    private String alertCode;
    private String deviceCode;
    private String workshopId;
    private String level;
    private String type;
    private String title;
    private String description;
    private String status;
    private String solution;
    private String acknowledgedBy;
    private String resolvedBy;
    private java.time.LocalDateTime occurredAt;
    private java.time.LocalDateTime acknowledgedAt;
    private java.time.LocalDateTime resolvedAt;
}
