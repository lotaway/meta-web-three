package com.metawebthree.product.domain.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("tb_product_attribute")
public class AttributeDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private Long productAttributeCategoryId;
    private String name;
    private Integer selectType;
    private Integer inputType;
    private String inputList;
    private Integer sort;
    private Integer filterType;
    private Integer searchType;
    private Integer relatedStatus;
    private Integer handAddStatus;
    private Integer type;
}
