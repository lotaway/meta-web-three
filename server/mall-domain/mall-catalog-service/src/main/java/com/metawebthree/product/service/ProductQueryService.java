package com.metawebthree.product.service;

import com.metawebthree.common.dto.ApiResponse;
import java.util.Map;

public interface ProductQueryService {
    ApiResponse<Map<String, Object>> getProductDetail(Long id, Long userId);
}
