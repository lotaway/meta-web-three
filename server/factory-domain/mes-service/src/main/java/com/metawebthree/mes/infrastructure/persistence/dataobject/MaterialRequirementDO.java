package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_material_requirement")
public class MaterialRequirementDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String requirementNo;
    private String workOrderNo;
    private String productCode;
    private String productName;
    private Integer quantity;
    private String bomVersion;
    private String status;
    private String warehouseId;
    private String workshopId;
    private String requirementType;
    private LocalDateTime requiredDate;
    private String createdBy;
    private String updatedBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}