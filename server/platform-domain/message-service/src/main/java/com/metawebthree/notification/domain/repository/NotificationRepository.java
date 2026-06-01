package com.metawebthree.notification.domain.repository;

import com.metawebthree.notification.domain.model.NotificationDO;
import java.util.List;

public interface NotificationRepository {
    NotificationDO save(NotificationDO notification);
    NotificationDO findById(Long id);
    List<NotificationDO> findByUserId(Long userId);
    List<NotificationDO> findByUserIdAndReadStatus(Long userId, Integer readStatus);
    List<NotificationDO> findByUserIdAndType(Long userId, String type);
    List<NotificationDO> findAll();
    boolean updateReadStatus(Long id, Integer readStatus);
    boolean deleteById(Long id);
}