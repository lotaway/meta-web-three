package com.metawebthree.promotion.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.promotion.infrastructure.persistence.model.AdvertiseRecord;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface AdvertiseMapper extends BaseMapper<AdvertiseRecord> {
}
