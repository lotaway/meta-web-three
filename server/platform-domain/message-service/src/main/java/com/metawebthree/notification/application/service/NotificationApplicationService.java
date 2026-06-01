package com.metawebthree.notification.application.service;

import com.metawebthree.notification.application.dto.NotificationDTO;
import com.metawebthree.notification.application.dto.NotificationSendDTO;
import com.metawebthree.notification.application.dto.TemplateConfigDTO;
import com.metawebthree.notification.domain.model.NotificationDO;
import com.metawebthree.notification.domain.model.NotificationChannel;
import com.metawebthree.notification.domain.model.ReadStatus;
import com.metawebthree.notification.domain.model.SendStatus;
import com.metawebthree.notification.domain.repository.NotificationRepository;
import com.metawebthree.notification.infrastructure.sender.NotificationSender;
import com.metawebthree.notification.infrastructure.sender.NotificationSenderFactory;
import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.enums.ResponseStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class NotificationApplicationService {

    private static final Logger logger = LoggerFactory.getLogger(NotificationApplicationService.class);

    private final NotificationRepository notificationRepository;
    private final NotificationSenderFactory senderFactory;

    public NotificationApplicationService(NotificationRepository notificationRepository,
                                           NotificationSenderFactory senderFactory) {
        this.notificationRepository = notificationRepository;
        this.senderFactory = senderFactory;
    }

    /**
     * Send notification
     */
    @Transactional
    public NotificationDTO send(NotificationSendDTO sendDTO) {
        NotificationDO notification = new NotificationDO();
        notification.setUserId(sendDTO.getUserId());
        notification.setNotificationType(sendDTO.getNotificationType());
        notification.setTitle(sendDTO.getTitle());
        notification.setContent(sendDTO.getContent());
        notification.setChannel(sendDTO.getChannel());
        notification.setSendStatus(SendStatus.PENDING.getCode());
        notification.setReadStatus(ReadStatus.UNREAD.getCode());
        notification.setSendTime(LocalDateTime.now());
        notification.setCreateTime(LocalDateTime.now());
        notification.setUpdateTime(LocalDateTime.now());
        notification.setExtraData(sendDTO.getExtraData());

        // Simulate sending notification through different channels
        boolean sent = sendNotification(notification);
        
        if (sent) {
            notification.setSendStatus(SendStatus.SENT.getCode());
        } else {
            notification.setSendStatus(SendStatus.FAILED.getCode());
        }

        notificationRepository.save(notification);
        return convertToDTO(notification);
    }

    /**
     * Send notification using template
     */
    @Transactional
    public NotificationDTO sendWithTemplate(String templateCode, Long userId, java.util.Map<String, String> params) {
        // Get template (simplified - in real implementation would query template)
        NotificationDO notification = new NotificationDO();
        notification.setUserId(userId);
        notification.setNotificationType("TEMPLATE");
        notification.setTitle("Notification: " + templateCode);
        notification.setContent("Template content for: " + templateCode);
        notification.setChannel(NotificationChannel.IN_SITE.getCode());
        notification.setSendStatus(SendStatus.SENT.getCode());
        notification.setReadStatus(ReadStatus.UNREAD.getCode());
        notification.setSendTime(LocalDateTime.now());
        notification.setCreateTime(LocalDateTime.now());
        notification.setUpdateTime(LocalDateTime.now());

        notificationRepository.save(notification);
        return convertToDTO(notification);
    }

    /**
     * Mark notification as read
     */
    @Transactional
    public NotificationDTO markAsRead(Long notificationId, Long userId) {
        NotificationDO notification = notificationRepository.findById(notificationId);
        if (notification == null) {
            throw new BusinessException(ResponseStatus.NOT_FOUND, "Notification not found");
        }

        if (!notification.getUserId().equals(userId)) {
            throw new BusinessException(ResponseStatus.FORBIDDEN, "Not authorized");
        }

        notification.setReadStatus(ReadStatus.READ.getCode());
        notification.setReadTime(LocalDateTime.now());
        notification.setUpdateTime(LocalDateTime.now());

        notificationRepository.save(notification);
        return convertToDTO(notification);
    }

    /**
     * Get notification by ID
     */
    public NotificationDTO getById(Long id) {
        NotificationDO notification = notificationRepository.findById(id);
        if (notification == null) {
            throw new BusinessException(ResponseStatus.NOT_FOUND, "Notification not found");
        }
        return convertToDTO(notification);
    }

    /**
     * Get all notifications for a user
     */
    public List<NotificationDTO> getByUserId(Long userId) {
        return notificationRepository.findByUserId(userId).stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    /**
     * Get unread notifications for a user
     */
    public List<NotificationDTO> getUnreadByUserId(Long userId) {
        return notificationRepository.findByUserIdAndReadStatus(userId, ReadStatus.UNREAD.getCode()).stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    /**
     * Get notifications by type
     */
    public List<NotificationDTO> getByType(Long userId, String type) {
        return notificationRepository.findByUserIdAndType(userId, type).stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    /**
     * Get all notifications (admin)
     */
    public List<NotificationDTO> getAll() {
        return notificationRepository.findAll().stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    /**
     * Configure message template
     */
    public void configureTemplate(TemplateConfigDTO configDTO) {
        // Template configuration would be saved to database
        // Simplified implementation
    }

    private boolean sendNotification(NotificationDO notification) {
        String channel = notification.getChannel();
        
        // Get appropriate sender based on channel
        NotificationSender sender = senderFactory.getSender(channel);
        
        if (sender == null) {
            logger.error("No sender available for channel: {}", channel);
            return false;
        }
        
        // Send notification using appropriate sender
        boolean sent = sender.send(
            notification.getUserId(),
            notification.getTitle(),
            notification.getContent(),
            notification.getExtraData()
        );
        
        if (sent) {
            logger.info("Notification sent successfully via channel: {} to user: {}", 
                channel, notification.getUserId());
        } else {
            logger.error("Failed to send notification via channel: {} to user: {}", 
                channel, notification.getUserId());
        }
        
        return sent;
    }

    private NotificationDTO convertToDTO(NotificationDO notification) {
        NotificationDTO dto = new NotificationDTO();
        dto.setId(notification.getId());
        dto.setUserId(notification.getUserId());
        dto.setNotificationType(notification.getNotificationType());
        dto.setTitle(notification.getTitle());
        dto.setContent(notification.getContent());
        dto.setChannel(notification.getChannel());
        dto.setSendStatus(notification.getSendStatus());
        dto.setReadStatus(notification.getReadStatus());
        dto.setExtraData(notification.getExtraData());

        // Set channel description
        if (notification.getChannel() != null) {
            for (NotificationChannel channel : NotificationChannel.values()) {
                if (channel.getCode().equals(notification.getChannel())) {
                    dto.setChannelDesc(channel.getDesc());
                    break;
                }
            }
        }

        // Set send status description
        if (notification.getSendStatus() != null) {
            for (SendStatus status : SendStatus.values()) {
                if (status.getCode().equals(notification.getSendStatus())) {
                    dto.setSendStatusDesc(status.getDesc());
                    break;
                }
            }
        }

        // Set read status description
        if (notification.getReadStatus() != null) {
            for (ReadStatus status : ReadStatus.values()) {
                if (status.getCode().equals(notification.getReadStatus())) {
                    dto.setReadStatusDesc(status.getDesc());
                    break;
                }
            }
        }

        // Format times
        if (notification.getSendTime() != null) {
            dto.setSendTime(notification.getSendTime().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
        }
        if (notification.getReadTime() != null) {
            dto.setReadTime(notification.getReadTime().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
        }

        return dto;
    }
}