package com.metawebthree.user.infrastructure.persistence.mapper;
import com.metawebthree.user.domain.model.*;

import com.github.yulichang.base.MPJBaseMapper;
import com.metawebthree.user.domain.model.TokenMappingDO;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

@Mapper
public interface TokenMappingMapper extends MPJBaseMapper<TokenMappingDO> {
    
    @Select("SELECT * FROM TokenMapping WHERE child_token = #{childToken} AND is_revoked = 0 AND expires_at > NOW()")
    TokenMappingDO findValidTokenMapping(@Param("childToken") String childToken);
    
    @Select("SELECT COUNT(*) FROM TokenMapping WHERE parent_token = #{parentToken} AND is_revoked = 0")
    Integer countActiveChildTokens(@Param("parentToken") String parentToken);
    
    @Select("SELECT * FROM TokenMapping WHERE parent_token = #{parentToken} AND user_id = #{userId} AND is_revoked = 0 AND expires_at > NOW()")
    TokenMappingDO findValidTokenMappingByParentAndUser(@Param("parentToken") String parentToken, @Param("userId") Long userId);
}