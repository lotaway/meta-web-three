package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_checklist_item")
public class ChecklistItemDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String itemCode;
    private String itemName;
    private String itemCategory;
    private String dataType;
    private String standardValue;
    private String upperLimit;
    private String lowerLimit;
    private String unit;
    private String checkMethod;
    private String abnormalJudgment;
    private Boolean isMandatory;
    private Integer sortOrder;
    private String status;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}