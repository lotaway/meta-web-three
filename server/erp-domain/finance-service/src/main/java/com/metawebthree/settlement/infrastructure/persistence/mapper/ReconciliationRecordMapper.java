package com.metawebthree.settlement.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.settlement.infrastructure.persistence.dataobject.ReconciliationRecordDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ReconciliationRecordMapper extends BaseMapper<ReconciliationRecordDO> {
}