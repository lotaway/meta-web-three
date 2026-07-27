package com.metawebthree.wallet.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.wallet.domain.entity.SolanaKeypair;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface SolanaKeypairMapper extends BaseMapper<SolanaKeypair> {
}
