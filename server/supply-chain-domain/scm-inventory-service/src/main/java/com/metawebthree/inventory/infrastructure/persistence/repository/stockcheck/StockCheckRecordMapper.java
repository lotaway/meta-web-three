package com.metawebthree.inventory.infrastructure.persistence.repository.stockcheck;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.inventory.domain.entity.stockcheck.StockCheckRecord;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface StockCheckRecordMapper extends BaseMapper<StockCheckRecord> {
}
