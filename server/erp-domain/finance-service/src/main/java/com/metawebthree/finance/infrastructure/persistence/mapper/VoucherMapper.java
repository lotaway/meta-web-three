package com.metawebthree.finance.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.finance.infrastructure.persistence.dataobject.VoucherDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface VoucherMapper extends BaseMapper<VoucherDO> {
}