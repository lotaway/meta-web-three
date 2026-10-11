package com.metawebthree.logistics.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("logistics_tracking_event")
public class TrackingEventDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String trackingNo;
    private String eventType;
    private String location;
    private String description;
    private String operator;
    private LocalDateTime occurredAt;
    private LocalDateTime createdAt;
}