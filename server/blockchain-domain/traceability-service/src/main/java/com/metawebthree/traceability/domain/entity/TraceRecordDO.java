package com.metawebthree.traceability.domain.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("trace_record")
public class TraceRecordDO {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long traceId;

    private String productId;

    private String productName;

    private String batchNumber;

    private String producer;

    private LocalDateTime productionTime;

    private Integer status;

    private LocalDateTime createTime;

    private LocalDateTime updateTime;
}