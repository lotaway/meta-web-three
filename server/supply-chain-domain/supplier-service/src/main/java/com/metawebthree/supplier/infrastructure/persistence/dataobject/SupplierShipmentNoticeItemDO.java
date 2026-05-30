package com.metawebthree.supplier.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("supplier_shipment_notice_item")
public class SupplierShipmentNoticeItemDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private Long noticeId;
    private String productCode;
    private String productName;
    private String unit;
    private BigDecimal quantity;
    private BigDecimal weight;
    private BigDecimal volume;
    private String batchNo;
    private LocalDateTime productionDate;
    private LocalDateTime expiryDate;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;
}