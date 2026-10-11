package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_status_config")
public class StatusConfigDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private Long machineId;
    private String statusCode;
    private String statusName;
    private String statusCategory;
    private Boolean isInitial;
    private Boolean isFinal;
    private String color;
    private String icon;
    private Integer sortOrder;
    private LocalDateTime createdAt;
}