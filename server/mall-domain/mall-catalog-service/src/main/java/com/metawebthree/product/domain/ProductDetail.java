package com.metawebthree.product.domain;

import lombok.Data;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;

@Data
public class ProductDetail {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private String detailMobileHtml;
    private Long reviewCount;
}
