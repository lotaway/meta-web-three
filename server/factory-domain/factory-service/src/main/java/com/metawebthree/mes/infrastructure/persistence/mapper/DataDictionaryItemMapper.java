package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.DataDictionaryItemDO;
import org.apache.ibatis.annotations.Mapper;

/**
 * 数据字典项 Mapper
 */
@Mapper
public interface DataDictionaryItemMapper extends BaseMapper<DataDictionaryItemDO> {
}