package com.metawebthree.order.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.order.domain.model.CompanyAddressDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CompanyAddressMapper extends BaseMapper<CompanyAddressDO> {
}
