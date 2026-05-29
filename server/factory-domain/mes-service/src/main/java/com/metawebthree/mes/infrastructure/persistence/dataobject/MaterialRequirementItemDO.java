package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_material_requirement_item")
public class MaterialRequirementItemDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private Long requirementId;
    private String materialCode;
    private String materialName;
    private String materialSpec;
    private String unitCode;
    private String unitName;
    private Double requiredQuantity;
    private Double issuedQuantity;
    private Double pendingQuantity;
    private Double scrapQuantity;
    private String locationId;
    private String batchNo;
    private String status;
    private String remark;
    private LocalDateTime requiredDate;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}