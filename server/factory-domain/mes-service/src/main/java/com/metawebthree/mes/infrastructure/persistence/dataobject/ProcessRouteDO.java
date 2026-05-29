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
    private LocalDateTime effectiveDate;   // 生效日期
    private LocalDateTime expiryDate;      // 失效日期
    private String steps;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}