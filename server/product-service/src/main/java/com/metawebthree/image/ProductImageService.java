package com.metawebthree.image;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.stereotype.Service;

@Service
public class ProductImageService extends ServiceImpl<ProductImageMapper, ProductImageDO> {

    ProductImageMapper productImageMapper;

    public ProductImageService(ProductImageMapper productImageMapper) {
        this.productImageMapper = productImageMapper;
    }

    public int create(Long productId, String imageId, String url) {
        ProductImageDO productImageDO = ProductImageDO.builder()
                .productId(productId)
                .imageId(imageId)
                .url(url)
                .build();
        return productImageMapper.insert(productImageDO);
    }
}
