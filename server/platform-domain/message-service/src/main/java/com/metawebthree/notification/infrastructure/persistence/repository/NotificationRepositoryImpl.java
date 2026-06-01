package com.metawebthree.notification.infrastructure.persistence.repository;

import com.metawebthree.notification.domain.model.NotificationDO;
import com.metawebthree.notification.domain.repository.NotificationRepository;
import com.metawebthree.notification.infrastructure.persistence.mapper.NotificationMapper;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public class NotificationRepositoryImpl implements NotificationRepository {

    private final NotificationMapper notificationMapper;

    public NotificationRepositoryImpl(NotificationMapper notificationMapper) {
        this.notificationMapper = notificationMapper;
    }

    @Override
    public NotificationDO save(NotificationDO notification) {
        if (notification.getId() == null) {
            notificationMapper.insert(notification);
        } else {
            notificationMapper.update(notification);
        }
        return notification;
    }

    @Override
    public NotificationDO findById(Long id) {
        return notificationMapper.selectById(id);
    }

    @Override
    public List<NotificationDO> findByUserId(Long userId) {
        return notificationMapper.selectByUserId(userId);
    }

    @Override
    public List<NotificationDO> findByUserIdAndReadStatus(Long userId, Integer readStatus) {
        return notificationMapper.selectByUserIdAndReadStatus(userId, readStatus);
    }

    @Override
    public List<NotificationDO> findByUserIdAndType(Long userId, String type) {
        return notificationMapper.selectByUserIdAndType(userId, type);
    }

    @Override
    public List<NotificationDO> findAll() {
        return notificationMapper.selectAll();
    }

    @Override
    public boolean updateReadStatus(Long id, Integer readStatus) {
        return notificationMapper.updateReadStatus(id, readStatus) > 0;
    }

    @Override
    public boolean deleteById(Long id) {
        return notificationMapper.deleteById(id) > 0;
    }
}