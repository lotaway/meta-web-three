package com.metawebthree.repository;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.entity.UserKYC;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface UserKYCRepository extends BaseMapper<UserKYC> {
    
    @Select("SELECT * FROM user_kyc WHERE user_id = #{userId}")
    UserKYC findByUserId(@Param("userId") Long userId);
    
    @Select("SELECT * FROM user_kyc WHERE user_id = #{userId} AND status = #{status}")
    UserKYC findByUserIdAndStatus(@Param("userId") Long userId, @Param("status") String status);
    
    @Select("SELECT * FROM user_kyc WHERE status = #{status}")
    List<UserKYC> findByStatus(@Param("status") String status);
    
    @Select("SELECT * FROM user_kyc WHERE level = #{level}")
    List<UserKYC> findByLevel(@Param("level") String level);
    
    @Select("SELECT * FROM user_kyc WHERE user_id = #{userId} AND status = 'APPROVED' ORDER BY level DESC LIMIT 1")
    UserKYC findHighestApprovedLevelByUserId(@Param("userId") Long userId);
    
    @Select("SELECT * FROM user_kyc WHERE user_id = #{userId} ORDER BY created_at DESC LIMIT 1")
    UserKYC findLatestByUserId(@Param("userId") Long userId);
    
    @Select("SELECT COUNT(*) FROM user_kyc WHERE user_id = #{userId} AND status = #{status}")
    boolean existsByUserIdAndStatus(@Param("userId") Long userId, @Param("status") String status);
    
    @Select("SELECT COUNT(*) FROM user_kyc WHERE status = 'PENDING'")
    Long countPendingKYC();
    
    @Select("SELECT * FROM user_kyc WHERE status = 'PENDING' ORDER BY created_at ASC")
    List<UserKYC> findPendingKYC();
} 