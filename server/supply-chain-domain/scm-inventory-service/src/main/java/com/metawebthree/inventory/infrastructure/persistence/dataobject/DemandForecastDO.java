package com.metawebthree.inventory.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
@TableName("demand_forecast")
public class DemandForecastDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String skuCode;
    private Long warehouseId;
    private Integer forecastPeriodDays;
    private Integer predictedQuantity;
    private Integer confidenceLevel;
    private String forecastMethod;
    private LocalDate forecastStartDate;
    private LocalDate forecastEndDate;
    private String status;
    private LocalDateTime generatedAt;
    private String notes;
}