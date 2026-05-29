package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_bom")
public class BomBillOfMaterialsDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String bomCode;
    private String productCode;
    private String productName;
    private String version;
    private String versionStatus;
    private LocalDateTime effectiveDate;
    private LocalDateTime expiryDate;
    private String bomType;
    private String processRouteId;
    private String description;
    private String status;
    private Integer itemCount;
    private String previousVersion;
    private String changeReason;
    private String createdBy;
    private String updatedBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}