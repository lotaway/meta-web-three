package com.metawebthree.promotion.infrastructure.persistence.mapper;

import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Mapper
public interface FlashOrderItemMapper {

    @Insert("INSERT INTO tb_order_item (id, order_id, product_id, product_name, product_pic, product_sku_id, product_quantity, product_price, promotion_name, promotion_amount, real_amount, created_at, updated_at) "
            + "VALUES (#{id}, #{orderId}, #{productId}, #{productName}, #{productPic}, #{skuId}, #{quantity}, #{unitPrice}, '闪购', #{promotionAmount}, #{realAmount}, #{now}, #{now})")
    int insertOrderItem(@Param("id") Long id, @Param("orderId") Long orderId,
                        @Param("productId") Long productId, @Param("productName") String productName,
                        @Param("productPic") String productPic,
                        @Param("skuId") Long skuId, @Param("quantity") Integer quantity,
                        @Param("unitPrice") BigDecimal unitPrice,
                        @Param("promotionAmount") BigDecimal promotionAmount,
                        @Param("realAmount") BigDecimal realAmount,
                        @Param("now") LocalDateTime now);
}
