package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_process_bom")
public class ProcessBomItemDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String processBomCode;
    private String productCode;
    private String processRouteId;
    private String processCode;
    private String processName;
    private String version;
    private String status;
    private String description;
    private String createdBy;
    private String updatedBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}