package com.metawebthree.groupbuying.domain.model;

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
@TableName("tb_group_buy_team")
public class GroupBuyTeamDO {
    @TableId(type = IdType.INPUT)
    private Long id;
    private Long activityId;
    private String teamNo;
    private Long leaderId;
    private Integer requiredQuantity;
    private Integer currentQuantity;
    private String status;
    private Long orderId;
    private Timestamp expireTime;
    private Timestamp createdAt;
    private Timestamp updatedAt;
}