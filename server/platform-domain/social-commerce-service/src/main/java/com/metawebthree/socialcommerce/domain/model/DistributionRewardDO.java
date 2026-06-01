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
@TableName("tb_distribution_reward")
public class DistributionRewardDO {
    @TableId(type = IdType.INPUT)
    private Long id;
    private Long referrerId;
    private Long buyerId;
    private Long orderId;
    private BigDecimal orderAmount;
    private BigDecimal commissionAmount;
    private Integer level;
    private String status;
    private Timestamp settledTime;
    private Timestamp createdAt;
    private Timestamp updatedAt;
}