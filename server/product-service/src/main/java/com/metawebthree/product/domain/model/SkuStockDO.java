package com.metawebthree.product.domain.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;

@Data
@TableName("tb_sku_stock")
public class SkuStockDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private Long productId;
    private String skuCode;
    private BigDecimal price;
    private Integer stock;
    private Integer lowStock;
    private String pic;
    private Integer sale;
    private BigDecimal promotionPrice;
    private Integer lockStock;
    private String spData;
}
