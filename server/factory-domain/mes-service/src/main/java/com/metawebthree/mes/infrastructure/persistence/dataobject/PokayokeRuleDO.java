package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_pokayoke_rule")
public class PokayokeRuleDO {
    
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    
    private String ruleCode;
    private String ruleName;
    private String ruleType;
    private String status;
    private String workstationId;
    private String processCode;
    private String productCode;
    private String triggerCondition;
    private String actionsJson;
    private Integer priority;
    private Boolean enabled;
    private String description;
    private Long createdBy;
    private LocalDateTime createdAt;
    private Long updatedBy;
    private LocalDateTime updatedAt;
    private Boolean deleted;
}