package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_dashboard_component")
public class DashboardComponentDO {
    
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    
    private String componentCode;
    private String componentName;
    private String componentType;
    private String configSchema;
    private String defaultConfig;
    private String icon;
    private String description;
    private Boolean enabled;
    private Integer sortOrder;
    private Long createdBy;
    private LocalDateTime createdAt;
    private Boolean deleted;
}