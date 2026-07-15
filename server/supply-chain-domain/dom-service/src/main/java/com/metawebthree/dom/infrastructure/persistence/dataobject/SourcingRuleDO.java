package com.metawebthree.dom.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("sourcing_rule")
public class SourcingRuleDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private String ruleName;
    private String ruleType;
    private Integer priority;
    private String warehouseIds;
    private String region;
    private Boolean enabled;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
