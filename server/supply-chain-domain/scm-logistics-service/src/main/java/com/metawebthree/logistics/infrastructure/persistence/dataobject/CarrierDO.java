package com.metawebthree.logistics.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("logistics_carrier")
public class CarrierDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String carrierCode;
    private String carrierName;
    private String carrierType;
    private String contact;
    private String phone;
    private String website;
    private String status;
    private BigDecimal baseFreight;
    private BigDecimal weightUnitPrice;
    private BigDecimal volumeUnitPrice;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}