package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.DataDictionaryDO;
import org.apache.ibatis.annotations.Mapper;

/**
 * 数据字典 Mapper
 */
@Mapper
public interface DataDictionaryMapper extends BaseMapper<DataDictionaryDO> {
}