package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_scada_device_command")
public class ScadaDeviceCommandDO {

    @TableId(type = IdType.AUTO)
    private Long id;
    private String commandCode;
    private String equipmentCode;
    private String commandType;
    private String payload;
    private String status;
    private String createdBy;
    private LocalDateTime createdAt;
    private LocalDateTime executedAt;
    private String resultMessage;
}
