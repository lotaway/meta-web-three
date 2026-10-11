package com.metawebthree.review.infrastructure.client;

import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.common.generated.rpc.*;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

@Component
public class ProductClient {

    @DubboReference(check = false, lazy = true)
    private ProductService productService;

    /**
     * Get product detail
     * @param productId product ID
     * @return ProductDetailProto
     */
    public ProductDetailProto getProductDetail(Long productId) {
        GetProductDetailRequest request = GetProductDetailRequest.newBuilder()
                .setProductId(productId)
                .build();
        
        try {
            GetProductDetailResponse response = productService.getProductDetail(request);
            return response.getProduct();
        } catch (Exception e) {
            throw new BusinessException(ResponseStatus.SYSTEM_ERROR, "Failed to query product: " + e.getMessage());
        }
    }
}