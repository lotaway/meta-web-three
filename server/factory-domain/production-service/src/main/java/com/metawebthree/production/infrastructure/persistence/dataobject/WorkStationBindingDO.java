package com.metawebthree.production.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("work_station_binding")
public class WorkStationBindingDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String workstationCode;
    private String bindingType;
    private String targetCode;
    private String targetName;
    private String targetType;
    private Integer quantity;
    private Boolean isPrimary;
    private String status;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}