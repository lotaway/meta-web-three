package com.metawebthree.supplier.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("supplier_shipment_notice")
public class SupplierShipmentNoticeDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String noticeNo;
    private String supplierCode;
    private String orderNo;
    private Long warehouseId;
    private LocalDateTime expectedShipmentDate;
    private LocalDateTime actualShipmentDate;
    private String shipmentMethod;
    private String carrierName;
    private String carrierContact;
    private String trackingNumber;
    private String vehicleNumber;
    private String driverName;
    private String driverPhone;
    private BigDecimal totalQuantity;
    private BigDecimal totalWeight;
    private BigDecimal totalVolume;
    private String status;
    private String remark;
    private String confirmer;
    private LocalDateTime confirmedAt;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;
}