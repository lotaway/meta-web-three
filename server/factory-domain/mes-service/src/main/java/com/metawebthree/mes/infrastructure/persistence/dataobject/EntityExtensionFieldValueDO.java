package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_entity_extension_field_value")
public class EntityExtensionFieldValueDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String entityType;
    private Long entityId;
    private String fieldCode;
    private String fieldValue;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}