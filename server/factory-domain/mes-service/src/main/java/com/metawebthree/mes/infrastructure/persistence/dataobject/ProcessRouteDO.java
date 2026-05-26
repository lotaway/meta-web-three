package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_process_route")
public class ProcessRouteDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String routeCode;
    private String routeName;
    private String productCode;
    private Integer version;
    private String status;
    private String steps; // JSON格式存储工序步骤
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}