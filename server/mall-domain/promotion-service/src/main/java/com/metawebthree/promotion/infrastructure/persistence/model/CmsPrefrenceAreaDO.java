package com.metawebthree.promotion.infrastructure.persistence.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("tb_cms_prefrence_area")
public class CmsPrefrenceAreaDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private String name;
    private String subTitle;
    private Integer sort;
    private Integer showStatus;
    private byte[] pic;
}
