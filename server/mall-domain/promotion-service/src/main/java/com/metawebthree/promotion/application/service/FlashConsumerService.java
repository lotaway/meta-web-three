package com.metawebthree.promotion.application.service;

import com.metawebthree.promotion.interfaces.web.dto.FlashOrderRequest;

import java.util.List;
import java.util.Map;

public interface FlashConsumerService {

    List<Map<String, Object>> getCurrentPromotions();

    List<Map<String, Object>> getCurrentSessionProducts(Long sessionId);

    Map<String, Object> getProductFlashInfo(Long productId);

    Long createFlashOrder(Long userId, FlashOrderRequest request);
}
