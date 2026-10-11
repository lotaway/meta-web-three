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
@TableName("tb_share_record")
public class ShareRecordDO {
    @TableId(type = IdType.INPUT)
    private Long id;
    private Long sharerId;
    private Long sharedItemId;
    private String itemType;
    private String shareChannel;
    private String shareUrl;
    private Integer clickCount;
    private Integer purchaseCount;
    private BigDecimal rewardAmount;
    private String status;
    private Timestamp createdAt;
    private Timestamp updatedAt;
}