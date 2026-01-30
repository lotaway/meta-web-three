package com.metawebthree.product.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.product.domain.model.ProductLimitsDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ProductLimitsMapper extends BaseMapper<ProductLimitsDO> {
}
