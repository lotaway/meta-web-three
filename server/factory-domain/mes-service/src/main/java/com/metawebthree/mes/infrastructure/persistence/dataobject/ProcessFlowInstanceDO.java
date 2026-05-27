package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_process_flow_instance")
public class ProcessFlowInstanceDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    
    private String instanceCode;
    private Long templateId;
    private String templateName;
    private String businessType;
    private String businessKey;
    private String currentNodeId;
    private String currentNodeName;
    private String status;
    private String flowData;
    private LocalDateTime startedAt;
    private Long startedBy;
    private LocalDateTime completedAt;
    private Long completedBy;
    private Long createdBy;
    private LocalDateTime createdAt;
    private Long updatedBy;
    private LocalDateTime updatedAt;
    private Boolean deleted;
}