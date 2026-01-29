package com.metawebthree.product;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.math.BigDecimal;

@Data
@TableName("tb_product_entity")
public class ProductEntityDO {
    @TableId(type = IdType.AUTO)
    private Integer id;
    private Integer productId;
    private String productArtno;
    private BigDecimal salePrice;
    private BigDecimal marketPrice;
    private Integer inventory;
    private String imageUrl;
    private Integer isUserDiscount;
    private BigDecimal cashBack;
    private Integer cashBackCycle;
    private Integer cycleUnit;
}
