package com.metawebthree.cs.infrastructure.persistence.mybatis;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.cs.domain.model.QuickReply;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface MybatisQuickReplyMapper extends BaseMapper<QuickReply> {
}
