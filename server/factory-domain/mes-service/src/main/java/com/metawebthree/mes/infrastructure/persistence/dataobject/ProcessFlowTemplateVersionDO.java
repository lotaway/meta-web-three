package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_process_flow_template_version")
public class ProcessFlowTemplateVersionDO {
    
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    
    private Long templateId;
    private Integer version;
    private String templateCode;
    private String templateName;
    private String description;
    private String flowData;
    private String status;
    private String changeDescription;
    private Boolean isCurrentVersion;
    private Long createdBy;
    private LocalDateTime createdAt;
    private Boolean deleted;
}