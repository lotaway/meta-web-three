package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_status_transition")
public class StatusTransitionDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private Long machineId;
    private String fromStatus;
    private String toStatus;
    private String transitionAction;
    private String conditionExpression;
    private String eventCode;
    private Boolean isAutoTransition;
    private Integer sortOrder;
    private LocalDateTime createdAt;
}