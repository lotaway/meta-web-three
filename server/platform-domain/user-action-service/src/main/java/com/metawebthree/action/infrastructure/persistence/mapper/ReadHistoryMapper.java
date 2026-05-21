package com.metawebthree.action.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.action.domain.model.ReadHistory;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ReadHistoryMapper extends BaseMapper<ReadHistory> {
}
