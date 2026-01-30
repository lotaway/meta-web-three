package com.metawebthree.product.domain.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("tb_product")
public class ProductDO {
    @TableId(type = IdType.AUTO)
    private Integer id;
    private String productNo;
    private String productName;
    private Integer creator;
    private LocalDateTime createTime;
    private String productRemark;
    private Integer isShelves;
    private String languageVersion;
}
