package com.metawebthree.dataanalysis.infrastructure.persistence.mapper;

import com.metawebthree.dataanalysis.domain.entity.UserProfileDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import java.util.List;

@Mapper
public interface UserProfileMapper {
    void insert(UserProfileDO record);
    void updateById(UserProfileDO record);
    UserProfileDO selectByUserId(@Param("userId") Long userId);
    List<UserProfileDO> selectAll();
    Long countNewUsers(@Param("startDate") String startDate, @Param("endDate") String endDate);
    Long countActiveUsers(@Param("startDate") String startDate, @Param("endDate") String endDate);
}