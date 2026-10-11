package com.metawebthree.socialcommerce.domain.model;

import java.math.BigDecimal;
import java.sql.Timestamp;

import com.baomidou.mybatisplus.annotation.TableName;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.IdType;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@TableName("tb_distribution_level")
public class DistributionLevelDO {
    @TableId(type = IdType.INPUT)
    private Long id;
    private String levelName;
    private Integer levelCode;
    private BigDecimal commissionRate;
    private Integer requiredSales;
    private Integer maxSubordinateLevel;
    private Integer status;
    private Timestamp createdAt;
    private Timestamp updatedAt;
}