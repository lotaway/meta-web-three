package com.metawebthree.product.service.impl;

import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.product.domain.ProductDetail;
import com.metawebthree.product.mapper.ProductMapper;
import com.metawebthree.product.service.ProductQueryService;
import org.springframework.stereotype.Service;
import java.util.Map;
import java.util.HashMap;

@Service
public class ProductQueryServiceImpl implements ProductQueryService {

    private final ProductMapper productMapper;

    public ProductQueryServiceImpl(ProductMapper productMapper) {
        this.productMapper = productMapper;
    }

    @Override
    public ApiResponse<Map<String, Object>> getProductDetail(Long id, Long userId) {
        ProductDetail product = productMapper
                .selectOne(Wrappers.lambdaQuery(ProductDetail.class).eq(ProductDetail::getId, id));
        if (product == null) {
            return ApiResponse.error(ResponseStatus.PRODUCT_NOT_FOUND);
        }

        Map<String, Object> data = new HashMap<>();
        data.put("id", product.getId());
        data.put("detailMobileHtml", product.getDetailMobileHtml());
        data.put("reviewCount", product.getReviewCount());
        data.put("favor", false);
        return ApiResponse.success(data);
    }
}
