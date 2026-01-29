package com.metawebthree.product;

import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("tb_product_limits")
public class ProductLimitsDO {
    @TableId
    private Integer productId;
    private Integer purchase;
    private Integer purchaseTimes;
    private String purchaseUnit;
}
