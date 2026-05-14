package com.metawebthree.promotion.infrastructure.persistence.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("tb_flash_promotion_log")
public class FlashPromotionLogDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private Long memberId;
    private Long productId;
    private String memberPhone;
    private String productName;
    private LocalDateTime subscribeTime;
    private LocalDateTime sendTime;
}
