package com.metawebthree.wallet.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.wallet.domain.entity.Wallet;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.Optional;

@Mapper
public interface WalletMapper extends BaseMapper<Wallet> {

    @Select("SELECT * FROM tb_wallet WHERE user_id = #{userId} AND chain_type = #{chainType} LIMIT 1")
    Optional<Wallet> findByUserIdAndChainType(@Param("userId") String userId, @Param("chainType") String chainType);
}
