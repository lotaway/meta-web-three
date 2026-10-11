package com.metawebthree.cart.infrastructure;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.cart.domain.CartItem;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CartItemMapper extends BaseMapper<CartItem> {
}
