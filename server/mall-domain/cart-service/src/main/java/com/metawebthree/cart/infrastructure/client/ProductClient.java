package com.metawebthree.cart.infrastructure.client;

import java.math.BigDecimal;

import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import com.metawebthree.cart.domain.ProductInfo;
import com.metawebthree.common.generated.rpc.GetProductDetailRequest;
import com.metawebthree.common.generated.rpc.GetProductDetailResponse;
import com.metawebthree.common.generated.rpc.ProductService;

@Component
public class ProductClient {

    @DubboReference(check = false, lazy = true)
    private ProductService productService;

    public ProductInfo getProductInfo(Long productId) {
        if (productId == null) {
            return null;
        }

        GetProductDetailRequest request = GetProductDetailRequest.newBuilder()
                .setProductId(productId)
                .build();

        GetProductDetailResponse response = productService.getProductDetail(request);
        
        if (response == null || !response.hasProduct()) {
            return null;
        }

        var product = response.getProduct();
        ProductInfo info = new ProductInfo();
        info.setName(product.getName());
        info.setPic(product.getPic());
        info.setSubTitle(product.getSubTitle());
        info.setPrice(BigDecimal.valueOf(product.getPrice()));
        info.setPictures(product.getPicturesList());
        
        return info;
    }
}