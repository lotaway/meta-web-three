package com.metawebthree.cart.application;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.cart.domain.CartItem;
import com.metawebthree.cart.domain.ProductInfo;
import com.metawebthree.cart.infrastructure.CartItemMapper;
import com.metawebthree.cart.infrastructure.client.ProductClient;
import com.metawebthree.cart.infrastructure.client.PromotionClient;
import com.metawebthree.cart.dto.CartItemDTO;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.util.Date;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class CartServiceImpl implements CartService {

    private final CartItemMapper cartItemMapper;
    private final ProductClient productClient;
    private final PromotionClient promotionClient;

    @Override
    public int add(CartItemDTO cartItemDTO) {
        enrichProductInfo(cartItemDTO);

        LambdaQueryWrapper<CartItem> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(CartItem::getMemberId, cartItemDTO.getMemberId())
                .eq(CartItem::getProductId, cartItemDTO.getProductId())
                .eq(CartItem::getDeleteStatus, 0);
        
        if (cartItemDTO.getProductSkuId() != null) {
            queryWrapper.eq(CartItem::getProductSkuId, cartItemDTO.getProductSkuId());
        }

        CartItem existingItem = cartItemMapper.selectOne(queryWrapper);

        if (existingItem == null) {
            CartItem cartItem = new CartItem();
            BeanUtils.copyProperties(cartItemDTO, cartItem);
            cartItem.setCreateDate(new Date());
            cartItem.setDeleteStatus(0);
            return cartItemMapper.insert(cartItem);
        } else {
            existingItem.setQuantity(existingItem.getQuantity() + cartItemDTO.getQuantity());
            existingItem.setModifyDate(new Date());
            return cartItemMapper.updateById(existingItem);
        }
    }

    private void enrichProductInfo(CartItemDTO cartItemDTO) {
        try {
            ProductInfo product = productClient.getProductInfo(cartItemDTO.getProductId());
            
            if (product != null) {
                cartItemDTO.setProductName(product.getName());
                cartItemDTO.setProductPic(product.getPic());
                cartItemDTO.setProductSubTitle(product.getSubTitle());
                cartItemDTO.setPrice(product.getPrice());
            }
        } catch (Exception e) {
            log.warn("Failed to enrich product info for productId: {}, error: {}", cartItemDTO.getProductId(), e.getMessage());
        }
    }

    @Override
    public List<CartItemDTO> list(Long memberId) {
        LambdaQueryWrapper<CartItem> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(CartItem::getMemberId, memberId)
                .eq(CartItem::getDeleteStatus, 0);
        
        List<CartItem> cartItems = cartItemMapper.selectList(queryWrapper);
        return cartItems.stream().map(item -> {
            CartItemDTO dto = new CartItemDTO();
            BeanUtils.copyProperties(item, dto);
            return dto;
        }).collect(Collectors.toList());
    }

    @Override
    public int updateQuantity(Long memberId, Long id, Integer quantity) {
        CartItem cartItem = new CartItem();
        cartItem.setId(id);
        cartItem.setQuantity(quantity);
        cartItem.setModifyDate(new Date());
        LambdaQueryWrapper<CartItem> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(CartItem::getMemberId, memberId).eq(CartItem::getId, id);
        return cartItemMapper.update(cartItem, queryWrapper);
    }

    @Override
    public int delete(Long memberId, List<Long> ids) {
        CartItem cartItem = new CartItem();
        cartItem.setDeleteStatus(1);
        LambdaQueryWrapper<CartItem> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(CartItem::getMemberId, memberId).in(CartItem::getId, ids);
        return cartItemMapper.update(cartItem, queryWrapper);
    }

    @Override
    public int clear(Long memberId) {
        CartItem cartItem = new CartItem();
        cartItem.setDeleteStatus(1);
        LambdaQueryWrapper<CartItem> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(CartItem::getMemberId, memberId);
        return cartItemMapper.update(cartItem, queryWrapper);
    }

    @Override
    public List<CartItemDTO> listWithPromotion(Long memberId) {
        // 获取购物车列表
        List<CartItemDTO> cartItems = list(memberId);
        
        // 查询促销信息并附加到购物车项
        for (CartItemDTO item : cartItems) {
            enrichPromotionInfo(item);
        }
        
        return cartItems;
    }

    private void enrichPromotionInfo(CartItemDTO item) {
        try {
            List<PromotionClient.PromotionInfo> promotions = promotionClient.getPromotionsByProductId(item.getProductId());

            if (promotions != null && !promotions.isEmpty()) {
                PromotionClient.PromotionInfo best = promotions.get(0);
                item.setPromotionTag(best.getPromotionTag());
                item.setPromotionType(best.getPromotionType());
                item.setDiscountAmount(best.getDiscountAmount());
            }
        } catch (Exception e) {
            log.warn("获取促销信息失败 - 商品ID: {}, 错误: {}", item.getProductId(), e.getMessage());
        }
    }

    @Override
    public void updateAttributes(Long memberId, Long id, CartItemDTO cartItem) {
        CartItem item = new CartItem();
        item.setId(id);
        if (cartItem.getProductSkuId() != null) {
            item.setProductSkuId(cartItem.getProductSkuId());
        }
        item.setModifyDate(new Date());

        LambdaQueryWrapper<CartItem> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(CartItem::getMemberId, memberId).eq(CartItem::getId, id);
        cartItemMapper.update(item, queryWrapper);
    }

    @Override
    public CartItemDTO getProductOptions(Long memberId, Long productId) {
        LambdaQueryWrapper<CartItem> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(CartItem::getMemberId, memberId)
                .eq(CartItem::getProductId, productId)
                .eq(CartItem::getDeleteStatus, 0);

        CartItem item = cartItemMapper.selectOne(queryWrapper);
        if (item == null) {
            return null;
        }
        CartItemDTO dto = new CartItemDTO();
        BeanUtils.copyProperties(item, dto);
        return dto;
    }
}