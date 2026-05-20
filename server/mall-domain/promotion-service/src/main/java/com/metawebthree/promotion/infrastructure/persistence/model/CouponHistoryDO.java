package com.metawebthree.promotion.infrastructure.persistence.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("tb_coupon_history")
public class CouponHistoryDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private Long couponId;
    private Long memberId;
    private String couponCode;
    private String memberNickname;
    private Integer getType;
    private LocalDateTime createTime;
    private Integer useStatus;
    private LocalDateTime useTime;
    private Long orderId;
    private String orderSn;
}
