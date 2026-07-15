package com.metawebthree.dom.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.dom.infrastructure.persistence.dataobject.DomOrderDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface DomOrderMapper extends BaseMapper<DomOrderDO> {
}
