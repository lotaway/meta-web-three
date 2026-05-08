package com.metawebthree.promotion.infrastructure.persistence.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;

@Data
@TableName("tb_flash_promotion_product_relation")
public class FlashPromotionProductRelationDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private Long flashPromotionId;
    private Long flashPromotionSessionId;
    private Long productId;
    private BigDecimal flashPromotionPrice;
    private Integer flashPromotionCount;
    private Integer flashPromotionLimit;
    private Integer sort;
}
