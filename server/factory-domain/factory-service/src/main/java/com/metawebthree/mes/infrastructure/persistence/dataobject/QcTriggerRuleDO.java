package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_qc_trigger_rule")
public class QcTriggerRuleDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String ruleCode;
    private String ruleName;
    private String triggerType;
    private String targetObject;
    private String conditionJson;
    private String inspectionType;
    private String inspectionPlanCode;
    private Boolean isEnabled;
    private Integer priority;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}