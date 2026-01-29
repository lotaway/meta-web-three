package com.metawebthree.product;

import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.math.BigDecimal;

@Data
@TableName("tb_product_stats")
public class ProductStatsDO {
    @TableId
    private Integer productId;
    private Integer commentNumber;
    private Integer scoreNumber;
    private BigDecimal scores;
}
