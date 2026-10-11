package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_template_checklist_item")
public class TemplateChecklistItemDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private Long templateId;
    private Long itemId;
    private Integer itemSequence;
    private Boolean isMandatory;
    private String defaultValue;
    private String abnormalJudgment;
    private Integer sortOrder;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}