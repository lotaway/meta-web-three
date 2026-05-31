package com.metawebthree.traceability.domain.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("trace_event")
public class TraceEventDO {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long traceId;

    private String eventType;

    private String description;

    private String location;

    private String operator;

    private LocalDateTime timestamp;

    private String extraData;

    private LocalDateTime createTime;
}