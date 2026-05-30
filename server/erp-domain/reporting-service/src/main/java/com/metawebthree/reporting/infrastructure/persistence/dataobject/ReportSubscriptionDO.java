package com.metawebthree.reporting.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * 报表订阅数据对象
 */
@Data
@TableName("report_subscription")
public class ReportSubscriptionDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private Long userId;
    private String userName;
    private String reportType;
    private String frequency;
    private String cronExpression;
    private String channel;
    private String recipient;
    private String webhookUrl;
    private Boolean enabled;
    private LocalDateTime nextSendTime;
    private LocalDateTime lastSendTime;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}