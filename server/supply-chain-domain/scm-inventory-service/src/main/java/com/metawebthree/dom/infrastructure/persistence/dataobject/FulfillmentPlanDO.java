package com.metawebthree.dom.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("fulfillment_plan")
public class FulfillmentPlanDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private Long domOrderId;
    private String domOrderNo;
    private Integer totalLines;
    private Integer fulfilledLines;
    private Integer partiallyFulfilledLines;
    private Integer unfulfilledLines;
    private String status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
