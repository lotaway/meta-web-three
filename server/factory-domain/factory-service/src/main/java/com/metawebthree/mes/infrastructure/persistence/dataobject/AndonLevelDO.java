package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.metawebthree.mes.domain.entity.AndonLevel;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_andon_level")
public class AndonLevelDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private String levelCode;
    private String levelName;
    private Integer levelValue;
    private Integer responseTimeoutMinutes;
    private Integer handlingTimeoutMinutes;
    private String colorCode;
    private String description;
    private AndonLevel.AndonLevelStatus status;
    private Integer sortOrder;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}