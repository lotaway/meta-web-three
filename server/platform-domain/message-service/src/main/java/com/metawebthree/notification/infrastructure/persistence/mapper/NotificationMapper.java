package com.metawebthree.notification.infrastructure.persistence.mapper;

import com.metawebthree.notification.domain.model.NotificationDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import java.util.List;

@Mapper
public interface NotificationMapper {
    int insert(NotificationDO record);
    int update(NotificationDO record);
    int deleteById(Long id);
    NotificationDO selectById(Long id);
    List<NotificationDO> selectByUserId(Long userId);
    List<NotificationDO> selectByUserIdAndReadStatus(@Param("userId") Long userId, @Param("readStatus") Integer readStatus);
    List<NotificationDO> selectByUserIdAndType(@Param("userId") Long userId, @Param("type") String type);
    List<NotificationDO> selectAll();
    int updateReadStatus(@Param("id") Long id, @Param("readStatus") Integer readStatus);
}