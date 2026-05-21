package com.metawebthree.action.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.action.domain.model.ProductCollection;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ProductCollectionMapper extends BaseMapper<ProductCollection> {
}
