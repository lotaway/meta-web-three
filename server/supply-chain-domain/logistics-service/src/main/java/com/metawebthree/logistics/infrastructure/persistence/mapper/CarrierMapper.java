package com.metawebthree.logistics.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.logistics.infrastructure.persistence.dataobject.CarrierDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CarrierMapper extends BaseMapper<CarrierDO> {
}