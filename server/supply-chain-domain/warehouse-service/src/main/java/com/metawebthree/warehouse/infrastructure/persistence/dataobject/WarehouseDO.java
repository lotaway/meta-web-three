package com.metawebthree.warehouse.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("warehouse")
public class WarehouseDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String warehouseCode;
    private String warehouseName;
    private String warehouseType;
    private String province;
    private String city;
    private String district;
    private String address;
    private String contact;
    private String phone;
    private Integer totalCapacity;
    private Integer usedCapacity;
    private Integer availableCapacity;
    private String status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;
}