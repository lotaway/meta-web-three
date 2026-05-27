package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_process_flow_template")
public class ProcessFlowTemplateDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    
    private String templateCode;
    private String templateName;
    private String description;
    private Integer version;
    private String flowData;
    private String status;
    private Long createdBy;
    private LocalDateTime createdAt;
    private Long updatedBy;
    private LocalDateTime updatedAt;
    private Boolean deleted;
}