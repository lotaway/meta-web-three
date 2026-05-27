package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_report_datasource")
public class ReportDatasourceDO {
    
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    
    private String datasourceCode;
    private String datasourceName;
    private String datasourceType;
    private String connectionConfig;
    private String description;
    private Boolean enabled;
    private Long createdBy;
    private LocalDateTime createdAt;
    private Long updatedBy;
    private LocalDateTime updatedAt;
    private Boolean deleted;
}