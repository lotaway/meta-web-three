package com.metawebthree.repository;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.entity.CreditProfile;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CreditProfileRepo extends BaseMapper<CreditProfile> {
}
