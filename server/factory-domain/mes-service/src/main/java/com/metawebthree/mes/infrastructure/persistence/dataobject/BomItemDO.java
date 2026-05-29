package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_bom_item")
public class BomItemDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private Long bomId;
    private String materialCode;
    private String materialName;
    private String materialSpec;
    private String unitCode;
    private String unitName;
    private Double quantity;
    private Double scrapRate;
    private Integer sequence;
    private String level;
    private String parentMaterialCode;
    private String itemType;
    private String position;
    private String remark;
    private String status;
    private Long substituteItemId;
    private String createdBy;
    private String updatedBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}