package com.metawebthree.traceability.domain.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("product_info")
public class ProductInfoDO {

    @TableId(type = IdType.INPUT)
    private String productId;

    private String productName;

    private String category;

    private String manufacturer;

    private String productionLocation;

    private LocalDateTime productionDate;

    private Boolean isActive;

    private LocalDateTime createTime;

    private LocalDateTime updateTime;
}