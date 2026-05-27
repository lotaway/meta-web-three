package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_process_node_type")
public class ProcessNodeTypeDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    
    private String nodeTypeCode;
    private String nodeTypeName;
    private String category;
    private String icon;
    private String configSchema;
    private String description;
    private Boolean enabled;
    private Integer sortOrder;
    private Long createdBy;
    private LocalDateTime createdAt;
    private Boolean deleted;
}