package com.metawebthree.order.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.order.domain.model.ReturnApplyDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ReturnApplyMapper extends BaseMapper<ReturnApplyDO> {
}
