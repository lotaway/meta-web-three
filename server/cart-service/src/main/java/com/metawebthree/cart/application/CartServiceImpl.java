package com.metawebthree.cart.application;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.cart.domain.CartItem;
import com.metawebthree.cart.infrastructure.CartItemMapper;
import com.metawebthree.common.dto.CartItemDTO;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Service;

import java.util.Date;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class CartServiceImpl implements CartService {

    private final CartItemMapper cartItemMapper;

    @Override
    public int add(CartItemDTO cartItemDTO) {
        // Check if item already exists in cart for this member and SKU
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
}
