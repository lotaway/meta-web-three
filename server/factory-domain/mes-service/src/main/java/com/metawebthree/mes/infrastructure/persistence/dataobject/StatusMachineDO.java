package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_status_machine")
public class StatusMachineDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String machineCode;
    private String machineName;
    private String entityType;
    private String description;
    private String initialStatus;
    private Boolean isDefault;
    private String status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}