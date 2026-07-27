package com.metawebthree.wallet.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.wallet.domain.entity.SolanaActivity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface SolanaActivityMapper extends BaseMapper<SolanaActivity> {
}
