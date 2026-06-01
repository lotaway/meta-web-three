package com.metawebthree.socialcommerce.domain.model;

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
@TableName("tb_distribution_relation")
public class DistributionRelationDO {
    @TableId(type = IdType.INPUT)
    private Long id;
    private Long userId;
    private Long referrerId;
    private Integer level;
    private Long rootReferrerId;
    private String status;
    private Timestamp bindTime;
    private Timestamp createdAt;
    private Timestamp updatedAt;
}