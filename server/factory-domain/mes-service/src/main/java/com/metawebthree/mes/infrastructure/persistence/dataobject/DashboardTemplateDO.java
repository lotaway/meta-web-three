package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_dashboard_template")
public class DashboardTemplateDO {
    
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    
    private String templateCode;
    private String templateName;
    private String templateType;
    private String description;
    private String layoutJson;
    private String componentsJson;
    private String datasourceConfig;
    private Integer refreshInterval;
    private String status;
    private Long createdBy;
    private LocalDateTime createdAt;
    private Long updatedBy;
    private LocalDateTime updatedAt;
    private Boolean deleted;
}