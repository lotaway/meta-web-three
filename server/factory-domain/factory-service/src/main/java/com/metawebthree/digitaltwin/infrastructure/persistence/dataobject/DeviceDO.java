package com.metawebthree.digitaltwin.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.TableName;
import com.metawebthree.common.DO.BaseDO;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Data
@NoArgsConstructor
@SuperBuilder
@EqualsAndHashCode(callSuper = false)
@TableName("devices")
public class DeviceDO extends BaseDO {
    private Long id;
    private String deviceCode;
    private String deviceName;
    private String deviceType;
    private String workshopId;
    private String productionLineId;
    private String status;
    private Double positionX;
    private Double positionY;
    private Double positionZ;
    private Double rotationY;
    private String ipAddress;
    private String macAddress;
    private String mqttTopic;
    private java.time.LocalDateTime lastHeartbeat;
}
