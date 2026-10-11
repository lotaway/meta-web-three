package com.metawebthree.rma.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("return_shipping")
public class ReturnShippingDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private Long rmaId;
    private String rmaNo;
    private String carrier;
    private String trackingNo;
    private String shippingMethod;
    private String originAddress;
    private String destinationAddress;
    private LocalDateTime shippingDate;
    private LocalDateTime estimatedArrivalDate;
    private String status;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
