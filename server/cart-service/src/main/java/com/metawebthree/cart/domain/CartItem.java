package com.metawebthree.cart.domain;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.io.Serializable;
import java.math.BigDecimal;
import java.util.Date;

@Data
@TableName("oms_cart_item")
public class CartItem implements Serializable {
    @TableId(type = IdType.AUTO)
    private Long id;
    private Long productId;
    private Long productSkuId;
    private Long memberId;
    private Integer quantity;
    private BigDecimal price;
    private String productPic;
    private String productName;
    private String productSubTitle;
    private String productSkuCode;
    private String memberNickname;
    private Date createDate;
    private Date modifyDate;
    private Integer deleteStatus;
    private Long productCategoryId;
    private String productBrand;
    private String productSn;
    private String productAttr;
}
