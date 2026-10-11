package com.metawebthree.mes.infrastructure.persistence.dataobject.labor;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_operator_skill")
public class OperatorSkillDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private Long operatorId;
    private String skillCode;
    private String skillName;
    private String skillLevel;
    private Boolean certified;
    private LocalDateTime certifiedAt;
    private LocalDateTime expiryAt;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
