package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ProcessMaterialDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ProcessMaterialMapper extends BaseMapper<ProcessMaterialDO> {
}