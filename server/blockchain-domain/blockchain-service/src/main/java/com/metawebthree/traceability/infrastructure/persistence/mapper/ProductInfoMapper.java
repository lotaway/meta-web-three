package com.metawebthree.traceability.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.traceability.domain.entity.ProductInfoDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

@Mapper
public interface ProductInfoMapper extends BaseMapper<ProductInfoDO> {

    @Select("SELECT * FROM product_info WHERE product_id = #{productId}")
    ProductInfoDO selectByProductId(String productId);
}