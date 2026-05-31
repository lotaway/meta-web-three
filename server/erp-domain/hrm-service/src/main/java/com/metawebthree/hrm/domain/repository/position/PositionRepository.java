package com.metawebthree.hrm.domain.repository.position;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.hrm.domain.entity.position.Position;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface PositionRepository extends BaseMapper<Position> {
}