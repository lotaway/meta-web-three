package com.metawebthree.product;

import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.math.BigDecimal;

@Data
@TableName("tb_goods_stats")
public class GoodsStatsDO {
    @TableId
    private Integer goodsId;
    private Integer commentNumber;
    private Integer scoreNumber;
    private BigDecimal scores;
}
