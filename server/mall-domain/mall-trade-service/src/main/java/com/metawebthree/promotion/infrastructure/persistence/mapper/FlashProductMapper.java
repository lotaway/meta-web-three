package com.metawebthree.promotion.infrastructure.persistence.mapper;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

@Mapper
public interface FlashProductMapper {

    @Select("SELECT name, pic FROM tb_product WHERE id = #{productId}")
    java.util.Map<String, Object> findProductInfo(@Param("productId") Long productId);
}
