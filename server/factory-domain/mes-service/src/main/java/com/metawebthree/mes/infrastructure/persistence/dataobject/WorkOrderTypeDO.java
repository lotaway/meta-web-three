package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_work_order_type")
public class WorkOrderTypeDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String typeCode;
    private String typeName;
    private String description;
    private String statusMachineCode;
    private String processRouteTemplate;
    private Boolean isDefault;
    private Integer sortOrder;
    private String status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}