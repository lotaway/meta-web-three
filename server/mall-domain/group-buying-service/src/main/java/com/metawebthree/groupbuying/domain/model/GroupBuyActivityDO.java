package com.metawebthree.groupbuying.domain.model;

import java.math.BigDecimal;
import java.sql.Timestamp;
import java.time.LocalDateTime;

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
@TableName("tb_group_buy_activity")
public class GroupBuyActivityDO {
    @TableId(type = IdType.INPUT)
    private Long id;
    private String activityName;
    private Long productId;
    private String productName;
    private BigDecimal singlePrice;
    private BigDecimal groupPrice;
    private Integer requiredQuantity;
    private Integer currentQuantity;
    private Integer status;
    private LocalDateTime startTime;
    private LocalDateTime endTime;
    private Integer validityHours;
    private Timestamp createdAt;
    private Timestamp updatedAt;
}