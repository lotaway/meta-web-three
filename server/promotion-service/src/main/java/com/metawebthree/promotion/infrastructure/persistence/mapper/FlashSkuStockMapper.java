package com.metawebthree.promotion.infrastructure.persistence.mapper;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Update;

@Mapper
public interface FlashSkuStockMapper {

    @Update("UPDATE tb_sku_stock SET stock = stock - #{quantity}, lock_stock = lock_stock + #{quantity} WHERE id = #{skuId} AND stock >= #{quantity}")
    int deductStock(@Param("skuId") Long skuId, @Param("quantity") Integer quantity);

    @Update("UPDATE tb_flash_promotion_product_relation SET flash_promotion_count = flash_promotion_count - #{quantity} WHERE id = #{relationId} AND flash_promotion_count >= #{quantity}")
    int deductFlashCount(@Param("relationId") Long relationId, @Param("quantity") Integer quantity);
}
