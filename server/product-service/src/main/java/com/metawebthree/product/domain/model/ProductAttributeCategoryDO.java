package com.metawebthree.product.domain.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("tb_product_attribute_category")
public class ProductAttributeCategoryDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private String name;
    private Integer attributeCount;
    private Integer paramCount;
}
