package com.metawebthree.cart.application;

import com.metawebthree.cart.dto.CartItemDTO;
import java.util.List;

public interface CartService {
    int add(CartItemDTO cartItem);
    List<CartItemDTO> list(Long memberId);
    int updateQuantity(Long memberId, Long id, Integer quantity);
    int delete(Long memberId, List<Long> ids);
    int clear(Long memberId);
    
    List<CartItemDTO> listWithPromotion(Long memberId);
    void updateAttributes(Long memberId, Long id, CartItemDTO cartItem);
    CartItemDTO getProductOptions(Long memberId, Long productId);
}
