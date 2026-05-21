package com.metawebthree.message.application;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.message.domain.model.Notification;
import com.metawebthree.message.infrastructure.persistence.mapper.NotificationMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;

@Service
@RequiredArgsConstructor
public class NotificationService {
    private final NotificationMapper notificationMapper;

    public void createNotification(Long userId, String title, String content,
            String type, String relatedId, String icon, String imageUrl) {
        Notification notification = Notification.builder()
                .userId(userId)
                .title(title)
                .content(content)
                .type(type)
                .relatedId(relatedId)
                .icon(icon)
                .imageUrl(imageUrl)
                .readStatus(0)
                .createTime(LocalDateTime.now())
                .build();
        notificationMapper.insert(notification);
    }

    public List<Notification> listByUser(Long userId, String type) {
        LambdaQueryWrapper<Notification> query = new LambdaQueryWrapper<Notification>()
                .eq(Notification::getUserId, userId)
                .orderByDesc(Notification::getCreateTime);
        if (type != null && !type.isEmpty()) {
            query.eq(Notification::getType, type);
        }
        return notificationMapper.selectList(query);
    }

    public void markAsRead(Long userId, Long notificationId) {
        Notification notification = notificationMapper.selectById(notificationId);
        if (notification != null && notification.getUserId().equals(userId)) {
            notification.setReadStatus(1);
            notificationMapper.updateById(notification);
        }
    }

    public void markAllAsRead(Long userId) {
        List<Notification> unread = notificationMapper.selectList(new LambdaQueryWrapper<Notification>()
                .eq(Notification::getUserId, userId)
                .eq(Notification::getReadStatus, 0));
        for (Notification n : unread) {
            n.setReadStatus(1);
            notificationMapper.updateById(n);
        }
    }

    public void deleteNotification(Long userId, Long notificationId) {
        Notification notification = notificationMapper.selectById(notificationId);
        if (notification != null && notification.getUserId().equals(userId)) {
            notificationMapper.deleteById(notificationId);
        }
    }

    public long getUnreadCount(Long userId) {
        return notificationMapper.selectCount(new LambdaQueryWrapper<Notification>()
                .eq(Notification::getUserId, userId)
                .eq(Notification::getReadStatus, 0));
    }
}
