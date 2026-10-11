package com.metawebthree.digitaltwin.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.digitaltwin.infrastructure.persistence.dataobject.WorkshopDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface WorkshopMapper extends BaseMapper<WorkshopDO> {
}
