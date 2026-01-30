package com.metawebthree.product.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.product.domain.model.ProductEntityDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ProductEntityMapper extends BaseMapper<ProductEntityDO> {
}
