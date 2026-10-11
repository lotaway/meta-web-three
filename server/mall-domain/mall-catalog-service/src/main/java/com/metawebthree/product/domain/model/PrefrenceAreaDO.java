package com.metawebthree.product.domain.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("tb_prefrence_area")
public class PrefrenceAreaDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private String name;
    private String subTitle;
    private String pic;
    private Integer sort;
    private Integer showStatus;
}
