package com.metawebthree.notification.interfaces.controller;

import com.metawebthree.notification.application.dto.NotificationDTO;
import com.metawebthree.notification.application.dto.NotificationSendDTO;
import com.metawebthree.notification.application.dto.TemplateConfigDTO;
import com.metawebthree.notification.application.service.NotificationApplicationService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/notification")
public class NotificationController {

    private final NotificationApplicationService notificationService;

    public NotificationController(NotificationApplicationService notificationService) {
        this.notificationService = notificationService;
    }

    /**
     * Send notification
     */
    @PostMapping("/send")
    public ResponseEntity<NotificationDTO> send(@RequestBody NotificationSendDTO sendDTO) {
        NotificationDTO result = notificationService.send(sendDTO);
        return ResponseEntity.ok(result);
    }

    /**
     * Send notification using template
     */
    @PostMapping("/send-template")
    public ResponseEntity<NotificationDTO> sendWithTemplate(
            @RequestParam String templateCode,
            @RequestParam Long userId,
            @RequestBody Map<String, String> params) {
        NotificationDTO result = notificationService.sendWithTemplate(templateCode, userId, params);
        return ResponseEntity.ok(result);
    }

    /**
     * Mark notification as read
     */
    @PostMapping("/{id}/read")
    public ResponseEntity<NotificationDTO> markAsRead(
            @PathVariable Long id,
            @RequestHeader("X-User-Id") Long userId) {
        NotificationDTO result = notificationService.markAsRead(id, userId);
        return ResponseEntity.ok(result);
    }

    /**
     * Get notification by ID
     */
    @GetMapping("/{id}")
    public ResponseEntity<NotificationDTO> getById(@PathVariable Long id) {
        NotificationDTO result = notificationService.getById(id);
        return ResponseEntity.ok(result);
    }

    /**
     * Get all notifications for a user
     */
    @GetMapping("/user/{userId}")
    public ResponseEntity<List<NotificationDTO>> getByUserId(@PathVariable Long userId) {
        List<NotificationDTO> result = notificationService.getByUserId(userId);
        return ResponseEntity.ok(result);
    }

    /**
     * Get unread notifications for a user
     */
    @GetMapping("/user/{userId}/unread")
    public ResponseEntity<List<NotificationDTO>> getUnread(@PathVariable Long userId) {
        List<NotificationDTO> result = notificationService.getUnreadByUserId(userId);
        return ResponseEntity.ok(result);
    }

    /**
     * Get notifications by type
     */
    @GetMapping("/user/{userId}/type/{type}")
    public ResponseEntity<List<NotificationDTO>> getByType(
            @PathVariable Long userId,
            @PathVariable String type) {
        List<NotificationDTO> result = notificationService.getByType(userId, type);
        return ResponseEntity.ok(result);
    }

    /**
     * Get all notifications (admin)
     */
    @GetMapping("/list")
    public ResponseEntity<List<NotificationDTO>> getAll() {
        List<NotificationDTO> result = notificationService.getAll();
        return ResponseEntity.ok(result);
    }

    /**
     * Configure message template
     */
    @PostMapping("/template/configure")
    public ResponseEntity<Void> configureTemplate(@RequestBody TemplateConfigDTO configDTO) {
        notificationService.configureTemplate(configDTO);
        return ResponseEntity.ok().build();
    }
}