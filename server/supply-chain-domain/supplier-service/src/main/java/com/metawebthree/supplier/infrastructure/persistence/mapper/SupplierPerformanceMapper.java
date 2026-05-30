package com.metawebthree.supplier.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.supplier.infrastructure.persistence.dataobject.SupplierPerformanceDO;
import org.apache.ibatis.annotations.Mapper;

/**
 * 供应商绩效评估 Mapper
 */
@Mapper
public interface SupplierPerformanceMapper extends BaseMapper<SupplierPerformanceDO> {
}