package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_scada_telemetry")
public class ScadaTelemetryRecordDO {

    @TableId(type = IdType.AUTO)
    private Long id;
    private String equipmentCode;
    private String topic;
    private String payload;
    private LocalDateTime collectTime;
    private LocalDateTime createdAt;
}
