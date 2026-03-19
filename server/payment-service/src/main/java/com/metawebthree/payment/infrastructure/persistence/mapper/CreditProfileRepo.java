package com.metawebthree.payment.infrastructure.persistence.mapper;

import com.github.yulichang.base.MPJBaseMapper;
import com.metawebthree.payment.domain.model.CreditProfile;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CreditProfileRepo extends MPJBaseMapper<CreditProfile> {
}
