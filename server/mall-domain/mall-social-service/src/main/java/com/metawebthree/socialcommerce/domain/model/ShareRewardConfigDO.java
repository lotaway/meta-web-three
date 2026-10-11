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
@TableName("tb_share_reward_config")
public class ShareRewardConfigDO {
    @TableId(type = IdType.INPUT)
    private Long id;
    private String configName;
    private Integer rewardType;
    private BigDecimal fixedAmount;
    private BigDecimal percentage;
    private Integer maxRewardCount;
    private BigDecimal maxRewardAmount;
    private Integer status;
    private Timestamp validFrom;
    private Timestamp validTo;
    private Timestamp createdAt;
    private Timestamp updatedAt;
}