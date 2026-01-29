package com.metawebthree.image;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
@TableName("tb_goods_gallery")
public class ProductImageDO {
    @TableId(type = IdType.AUTO)
    private Integer id;
    private Integer goodsId;
    private String imageUrl;
    private Integer sortOrder;
}
