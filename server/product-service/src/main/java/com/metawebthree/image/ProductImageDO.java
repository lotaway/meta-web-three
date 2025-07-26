package com.metawebthree.image;

import com.baomidou.mybatisplus.annotation.TableName;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
@TableName("product_image")
public class ProductImageDO {
    private Long id;
    private String imageId;
    private Long productId;
    private String url;
}   
