package com.metawebthree.warehouse.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("stocktake_order")
public class StocktakeOrderDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String orderNo;
    private String type;
    private Long warehouseId;
    private String warehouseName;
    private Long locationId;
    private String locationName;
    private String status;
    private String operator;
    private LocalDateTime plannedDate;
    private LocalDateTime startDate;
    private LocalDateTime endDate;
    private Integer totalSkuCount;
    private Integer checkedSkuCount;
    private Integer discrepancyCount;
    private BigDecimal totalDiscrepancyAmount;
    private String remark;
    private String createdBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;
}
