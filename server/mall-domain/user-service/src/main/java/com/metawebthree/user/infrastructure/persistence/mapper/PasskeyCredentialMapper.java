package com.metawebthree.user.infrastructure.persistence.mapper;

import com.github.yulichang.base.MPJBaseMapper;
import com.metawebthree.user.domain.model.PasskeyCredentialDO;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface PasskeyCredentialMapper extends MPJBaseMapper<PasskeyCredentialDO> {
    @Select("SELECT * FROM PasskeyCredential WHERE user_id = #{userId} AND is_revoked = 0")
    List<PasskeyCredentialDO> selectByUserId(Long userId);

    @Select("SELECT * FROM PasskeyCredential WHERE credential_id = #{credentialId} AND is_revoked = 0")
    PasskeyCredentialDO selectByCredentialId(String credentialId);
}
