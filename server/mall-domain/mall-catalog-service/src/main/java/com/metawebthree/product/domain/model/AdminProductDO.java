package com.metawebthree.product.domain.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;

@Data
@TableName("tb_product")
public class AdminProductDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private Long brandId;
    private Long productCategoryId;
    private String name;
    private String pic;
    private String productSn;
    private Integer deleteStatus;
    private Integer publishStatus;
    private Integer newStatus;
    private Integer recommandStatus;
    private Integer verifyStatus;
    private Integer sort;
    private Integer sale;
    private BigDecimal price;
    private BigDecimal promotionPrice;
    private Integer stock;
    private Integer lowStock;
    private String unit;
}
