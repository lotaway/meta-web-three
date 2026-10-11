package com.metawebthree.warehouse.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("inbound_order")
public class InboundOrderDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String orderNo;
    private String inboundType;
    private Long warehouseId;
    private String supplierCode;
    private String status;
    private String remark;
    private String operator;
    private LocalDateTime planArrivalTime;
    private LocalDateTime actualArrivalTime;
    private LocalDateTime completedAt;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;
}