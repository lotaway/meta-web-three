package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_material_substitute_item")
public class MaterialSubstituteItemDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private Long substituteGroupId;
    private String materialCode;
    private String materialName;
    private String materialSpec;
    private String unitCode;
    private String unitName;
    private Integer priority;
    private Double conversionRate;
    private String conversionUnit;
    private String reason;
    private LocalDateTime effectiveDate;
    private LocalDateTime expiryDate;
    private String status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}