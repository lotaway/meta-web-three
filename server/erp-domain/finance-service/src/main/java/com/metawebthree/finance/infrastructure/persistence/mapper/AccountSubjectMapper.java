package com.metawebthree.finance.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.finance.infrastructure.persistence.dataobject.AccountSubjectDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface AccountSubjectMapper extends BaseMapper<AccountSubjectDO> {
}