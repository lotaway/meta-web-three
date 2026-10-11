package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_bom_version")
public class BomVersionDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String productCode;
    private String productName;
    private Long bomId;
    private String version;
    private String versionStatus;
    private LocalDateTime effectiveDate;
    private LocalDateTime expiryDate;
    private String changeType;
    private String changeReason;
    private String changedBy;
    private LocalDateTime changedAt;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}