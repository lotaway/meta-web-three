package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.metawebthree.mes.domain.entity.AndonType;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_andon_type")
public class AndonTypeDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String typeCode;
    private String typeName;
    private AndonType.AndonCategory category;
    private String description;
    private Boolean requirePhoto;
    private Boolean requireConfirm;
    private Integer defaultEscalationMinutes;
    private String defaultProcessTemplate;
    private AndonType.AndonStatus status;
    private Integer sortOrder;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}