package com.metawebthree.inventory.infrastructure.persistence.repository.stockcheck;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.inventory.domain.entity.stockcheck.StockCheckDiff;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface StockCheckDiffMapper extends BaseMapper<StockCheckDiff> {
}
