package com.metawebthree.product;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("tb_goods")
public class GoodsDO {
    @TableId(type = IdType.AUTO)
    private Integer id;
    private String goodsNo;
    private String goodsName;
    private Integer creator;
    private LocalDateTime createTime;
    private String goodsRemark;
    private Integer isShelves;
    private String languageVersion;
}
