package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.metawebthree.mes.domain.entity.WorkOrderSplitRule;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_work_order_split_rule")
public class WorkOrderSplitRuleDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private String ruleName;
    private String ruleCode;
    private String splitType;
    private String description;
    private Boolean enabled;
    private Integer splitQuantity;
    private Integer maxChildOrders;
    private String bomId;
    private String processRouteId;
    private String conditions;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}