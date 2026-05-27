package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ProcessNodeTypeDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ProcessNodeTypeMapper extends BaseMapper<ProcessNodeTypeDO> {
}