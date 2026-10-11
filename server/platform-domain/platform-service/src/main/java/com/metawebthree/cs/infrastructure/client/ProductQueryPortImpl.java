package com.metawebthree.cs.infrastructure.client;

import java.util.Optional;

import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import com.metawebthree.common.generated.rpc.*;
import com.metawebthree.cs.domain.ports.ProductQueryPort;

import lombok.extern.slf4j.Slf4j;

@Component
@Slf4j
public class ProductQueryPortImpl implements ProductQueryPort {

    @DubboReference(check = false, lazy = true)
    private ProductService productService;

    @Override
    public Optional<String> findProductName(Long productId) {
        try {
            GetProductDetailRequest request = GetProductDetailRequest.newBuilder()
                    .setProductId(productId)
                    .build();
            GetProductDetailResponse response = productService.getProductDetail(request);
            if (response != null && response.hasProduct()) {
                return Optional.of(response.getProduct().getName());
            }
        } catch (Exception e) {
            log.error("Query product name failed, productId: {}, error: {}", productId, e.getMessage());
        }
        return Optional.empty();
    }

    @Override
    public Optional<String> findProductJson(Long productId) {
        try {
            GetProductDetailRequest request = GetProductDetailRequest.newBuilder()
                    .setProductId(productId)
                    .build();
            GetProductDetailResponse response = productService.getProductDetail(request);
            if (response != null && response.hasProduct()) {
                return Optional.of(response.getProduct().toString());
            }
        } catch (Exception e) {
            log.error("Query product json failed, productId: {}, error: {}", productId, e.getMessage());
        }
        return Optional.empty();
    }
}