package com.metawebthree.logistics.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.logistics.infrastructure.persistence.dataobject.TrackingEventDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface TrackingEventMapper extends BaseMapper<TrackingEventDO> {
}