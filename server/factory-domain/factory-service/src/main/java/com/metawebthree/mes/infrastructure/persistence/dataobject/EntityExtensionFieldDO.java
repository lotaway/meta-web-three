package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_entity_extension_field")
public class EntityExtensionFieldDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String entityType;
    private String fieldCode;
    private String fieldName;
    private String fieldType;
    private String defaultValue;
    private Boolean required;
    private Boolean isUnique;
    private String validationRule;
    private Boolean listVisible;
    private Boolean searchable;
    private Integer sortOrder;
    private String fieldGroup;
    private String referenceType;
    private String referenceEntity;
    private String status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}