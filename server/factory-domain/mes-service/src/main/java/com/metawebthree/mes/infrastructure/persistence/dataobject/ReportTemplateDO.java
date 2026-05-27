package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_report_template")
public class ReportTemplateDO {
    
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    
    private String templateCode;
    private String templateName;
    private String reportType;
    private String description;
    private String configJson;
    private String dataSourceType;
    private String dataSourceConfig;
    private String querySql;
    private String parametersJson;
    private String status;
    private Integer version;
    private Long createdBy;
    private LocalDateTime createdAt;
    private Long updatedBy;
    private LocalDateTime updatedAt;
    private Boolean deleted;
}