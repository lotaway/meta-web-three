package com.metawebthree.supplier.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.supplier.infrastructure.persistence.entity.SupplierEntity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface SupplierMapper extends BaseMapper<SupplierEntity> {
}