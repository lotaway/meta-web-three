package com.metawebthree.product;

import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("tb_goods_limits")
public class GoodsLimitsDO {
    @TableId
    private Integer goodsId;
    private Integer purchase;
    private Integer purchaseTimes;
    private String purchaseUnit;
}
