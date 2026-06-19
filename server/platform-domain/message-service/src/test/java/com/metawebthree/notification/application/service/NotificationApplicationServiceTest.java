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
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class NotificationApplicationServiceTest {

    @Mock
    private NotificationRepository notificationRepository;

    @Mock
    private NotificationSenderFactory senderFactory;

    @Mock
    private NotificationSender notificationSender;

    private NotificationApplicationService service;

    @BeforeEach
    void setUp() {
        service = new NotificationApplicationService(notificationRepository, senderFactory);
    }

    // ==================== Send Notification Tests ====================

    @Test
    void send_shouldReturnNotificationDTO_whenSenderSucceeds() {
        // Arrange
        NotificationSendDTO sendDTO = new NotificationSendDTO();
        sendDTO.setUserId(1L);
        sendDTO.setNotificationType("SYSTEM");
        sendDTO.setTitle("Test Notification");
        sendDTO.setContent("Test Content");
        sendDTO.setChannel(NotificationChannel.IN_SITE.getCode());
        sendDTO.setExtraData("{}");

        when(senderFactory.getSender(anyString())).thenReturn(notificationSender);
        when(notificationSender.send(anyLong(), anyString(), anyString(), anyString())).thenReturn(true);
        when(notificationRepository.save(any(NotificationDO.class))).thenAnswer(invocation -> {
            NotificationDO notification = invocation.getArgument(0);
            notification.setId(1L);
            return notification;
        });

        // Act
        NotificationDTO result = service.send(sendDTO);

        // Assert
        assertNotNull(result);
        assertEquals(1L, result.getUserId());
        assertEquals("Test Notification", result.getTitle());
        assertEquals(SendStatus.SENT.getCode(), result.getSendStatus());
        verify(notificationRepository).save(any(NotificationDO.class));
    }

    @Test
    void send_shouldReturnFailedStatus_whenSenderFails() {
        // Arrange
        NotificationSendDTO sendDTO = new NotificationSendDTO();
        sendDTO.setUserId(1L);
        sendDTO.setNotificationType("SYSTEM");
        sendDTO.setTitle("Test Notification");
        sendDTO.setContent("Test Content");
        sendDTO.setChannel(NotificationChannel.IN_SITE.getCode());

        when(senderFactory.getSender(anyString())).thenReturn(notificationSender);
        when(notificationSender.send(anyLong(), anyString(), anyString(), anyString())).thenReturn(false);
        when(notificationRepository.save(any(NotificationDO.class))).thenAnswer(invocation -> {
            NotificationDO notification = invocation.getArgument(0);
            notification.setId(1L);
            return notification;
        });

        // Act
        NotificationDTO result = service.send(sendDTO);

        // Assert
        assertNotNull(result);
        assertEquals(SendStatus.FAILED.getCode(), result.getSendStatus());
    }

    @Test
    void send_shouldReturnFailedStatus_whenSenderIsNull() {
        // Arrange
        NotificationSendDTO sendDTO = new NotificationSendDTO();
        sendDTO.setUserId(1L);
        sendDTO.setNotificationType("SYSTEM");
        sendDTO.setTitle("Test Notification");
        sendDTO.setContent("Test Content");
        sendDTO.setChannel("UNKNOWN_CHANNEL");

        when(senderFactory.getSender("UNKNOWN_CHANNEL")).thenReturn(null);
        when(notificationRepository.save(any(NotificationDO.class))).thenAnswer(invocation -> {
            NotificationDO notification = invocation.getArgument(0);
            notification.setId(1L);
            return notification;
        });

        // Act
        NotificationDTO result = service.send(sendDTO);

        // Assert
        assertNotNull(result);
        assertEquals(SendStatus.FAILED.getCode(), result.getSendStatus());
    }

    // ==================== Send With Template Tests ====================

    @Test
    void sendWithTemplate_shouldReturnNotificationDTO_whenTemplateProvided() {
        // Arrange
        String templateCode = "ORDER_CONFIRM";
        Long userId = 1L;
        Map<String, String> params = Map.of("orderId", "12345");

        when(notificationRepository.save(any(NotificationDO.class))).thenAnswer(invocation -> {
            NotificationDO notification = invocation.getArgument(0);
            notification.setId(1L);
            return notification;
        });

        // Act
        NotificationDTO result = service.sendWithTemplate(templateCode, userId, params);

        // Assert
        assertNotNull(result);
        assertEquals(userId, result.getUserId());
        assertEquals("TEMPLATE", result.getNotificationType());
        assertEquals(NotificationChannel.IN_SITE.getCode(), result.getChannel());
    }

    // ==================== Mark As Read Tests ====================

    @Test
    void markAsRead_shouldReturnNotificationDTO_whenNotificationExists() {
        // Arrange
        Long notificationId = 1L;
        Long userId = 1L;
        NotificationDO notification = new NotificationDO();
        notification.setId(notificationId);
        notification.setUserId(userId);
        notification.setReadStatus(ReadStatus.UNREAD.getCode());

        when(notificationRepository.findById(notificationId)).thenReturn(notification);
        when(notificationRepository.save(any(NotificationDO.class))).thenReturn(notification);

        // Act
        NotificationDTO result = service.markAsRead(notificationId, userId);

        // Assert
        assertNotNull(result);
        assertEquals(ReadStatus.READ.getCode(), result.getReadStatus());
    }

    @Test
    void markAsRead_shouldThrowException_whenNotificationNotFound() {
        // Arrange
        Long notificationId = 999L;
        Long userId = 1L;

        when(notificationRepository.findById(notificationId)).thenReturn(null);

        // Act & Assert
        BusinessException exception = assertThrows(BusinessException.class, 
            () -> service.markAsRead(notificationId, userId));
        assertEquals(ResponseStatus.NOT_FOUND, exception.getStatus());
    }

    @Test
    void markAsRead_shouldThrowException_whenUserNotAuthorized() {
        // Arrange
        Long notificationId = 1L;
        Long userId = 1L;
        Long differentUserId = 2L;
        NotificationDO notification = new NotificationDO();
        notification.setId(notificationId);
        notification.setUserId(differentUserId);

        when(notificationRepository.findById(notificationId)).thenReturn(notification);

        // Act & Assert
        BusinessException exception = assertThrows(BusinessException.class, 
            () -> service.markAsRead(notificationId, userId));
        assertEquals(ResponseStatus.FORBIDDEN, exception.getStatus());
    }

    // ==================== Get By ID Tests ====================

    @Test
    void getById_shouldReturnNotificationDTO_whenNotificationExists() {
        // Arrange
        Long notificationId = 1L;
        NotificationDO notification = new NotificationDO();
        notification.setId(notificationId);
        notification.setUserId(1L);
        notification.setTitle("Test");
        notification.setContent("Content");
        notification.setChannel(NotificationChannel.IN_SITE.getCode());
        notification.setSendStatus(SendStatus.SENT.getCode());
        notification.setReadStatus(ReadStatus.UNREAD.getCode());

        when(notificationRepository.findById(notificationId)).thenReturn(notification);

        // Act
        NotificationDTO result = service.getById(notificationId);

        // Assert
        assertNotNull(result);
        assertEquals(notificationId, result.getId());
        assertEquals("Test", result.getTitle());
    }

    @Test
    void getById_shouldThrowException_whenNotificationNotFound() {
        // Arrange
        Long notificationId = 999L;
        when(notificationRepository.findById(notificationId)).thenReturn(null);

        // Act & Assert
        BusinessException exception = assertThrows(BusinessException.class, 
            () -> service.getById(notificationId));
        assertEquals(ResponseStatus.NOT_FOUND, exception.getStatus());
    }

    // ==================== Get By User ID Tests ====================

    @Test
    void getByUserId_shouldReturnListOfNotifications() {
        // Arrange
        Long userId = 1L;
        NotificationDO notification1 = new NotificationDO();
        notification1.setId(1L);
        notification1.setUserId(userId);
        notification1.setChannel(NotificationChannel.IN_SITE.getCode());
        notification1.setSendStatus(SendStatus.SENT.getCode());
        notification1.setReadStatus(ReadStatus.UNREAD.getCode());

        NotificationDO notification2 = new NotificationDO();
        notification2.setId(2L);
        notification2.setUserId(userId);
        notification2.setChannel(NotificationChannel.SMS.getCode());
        notification2.setSendStatus(SendStatus.SENT.getCode());
        notification2.setReadStatus(ReadStatus.READ.getCode());

        when(notificationRepository.findByUserId(userId)).thenReturn(List.of(notification1, notification2));

        // Act
        List<NotificationDTO> results = service.getByUserId(userId);

        // Assert
        assertNotNull(results);
        assertEquals(2, results.size());
    }

    @Test
    void getByUserId_shouldReturnEmptyList_whenNoNotifications() {
        // Arrange
        Long userId = 1L;
        when(notificationRepository.findByUserId(userId)).thenReturn(List.of());

        // Act
        List<NotificationDTO> results = service.getByUserId(userId);

        // Assert
        assertNotNull(results);
        assertTrue(results.isEmpty());
    }

    // ==================== Get Unread By User ID Tests ====================

    @Test
    void getUnreadByUserId_shouldReturnListOfUnreadNotifications() {
        // Arrange
        Long userId = 1L;
        NotificationDO notification = new NotificationDO();
        notification.setId(1L);
        notification.setUserId(userId);
        notification.setReadStatus(ReadStatus.UNREAD.getCode());
        notification.setChannel(NotificationChannel.IN_SITE.getCode());
        notification.setSendStatus(SendStatus.SENT.getCode());

        when(notificationRepository.findByUserIdAndReadStatus(userId, ReadStatus.UNREAD.getCode()))
            .thenReturn(List.of(notification));

        // Act
        List<NotificationDTO> results = service.getUnreadByUserId(userId);

        // Assert
        assertNotNull(results);
        assertEquals(1, results.size());
        assertEquals(ReadStatus.UNREAD.getCode(), results.get(0).getReadStatus());
    }

    // ==================== Get By Type Tests ====================

    @Test
    void getByType_shouldReturnListOfNotificationsByType() {
        // Arrange
        Long userId = 1L;
        String type = "SYSTEM";
        NotificationDO notification = new NotificationDO();
        notification.setId(1L);
        notification.setUserId(userId);
        notification.setNotificationType(type);
        notification.setChannel(NotificationChannel.IN_SITE.getCode());
        notification.setSendStatus(SendStatus.SENT.getCode());
        notification.setReadStatus(ReadStatus.UNREAD.getCode());

        when(notificationRepository.findByUserIdAndType(userId, type)).thenReturn(List.of(notification));

        // Act
        List<NotificationDTO> results = service.getByType(userId, type);

        // Assert
        assertNotNull(results);
        assertEquals(1, results.size());
        assertEquals(type, results.get(0).getNotificationType());
    }

    // ==================== Get All Tests ====================

    @Test
    void getAll_shouldReturnListOfAllNotifications() {
        // Arrange
        NotificationDO notification1 = new NotificationDO();
        notification1.setId(1L);
        notification1.setUserId(1L);
        notification1.setChannel(NotificationChannel.IN_SITE.getCode());
        notification1.setSendStatus(SendStatus.SENT.getCode());
        notification1.setReadStatus(ReadStatus.UNREAD.getCode());

        NotificationDO notification2 = new NotificationDO();
        notification2.setId(2L);
        notification2.setUserId(2L);
        notification2.setChannel(NotificationChannel.EMAIL.getCode());
        notification2.setSendStatus(SendStatus.SENT.getCode());
        notification2.setReadStatus(ReadStatus.READ.getCode());

        when(notificationRepository.findAll()).thenReturn(List.of(notification1, notification2));

        // Act
        List<NotificationDTO> results = service.getAll();

        // Assert
        assertNotNull(results);
        assertEquals(2, results.size());
    }

    // ==================== Configure Template Tests ====================

    @Test
    void configureTemplate_shouldCompleteWithoutException() {
        // Arrange
        TemplateConfigDTO configDTO = new TemplateConfigDTO();
        configDTO.setTemplateCode("TEST_TEMPLATE");
        configDTO.setTemplateName("Test Template");
        configDTO.setNotificationType("SYSTEM");
        configDTO.setTitleTemplate("Test Title");
        configDTO.setContentTemplate("Test Content");
        configDTO.setChannel("IN_SITE");
        configDTO.setEnabled(true);

        // Act & Assert - No exception should be thrown
        assertDoesNotThrow(() -> service.configureTemplate(configDTO));
    }
}