package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_work_report")
public class WorkReportDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String reportNo;
    private Long taskId;
    private String taskNo;
    private Long workOrderId;
    private String workOrderNo;
    private Long workstationId;
    private String workstationName;
    private String processCode;
    private String processName;
    private Integer stepNo;
    private String operatorId;
    private String operatorName;
    private LocalDateTime reportTime;
    private Integer quantity;
    private Integer qualifiedQuantity;
    private Integer defectiveQuantity;
    private Integer durationMinutes;
    private String parameterValuesJson;
    private String remarks;
    private String status;
    private String createdBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}