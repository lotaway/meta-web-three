package com.metawebthree.message.interfaces.web;

import com.metawebthree.message.application.NotificationService;
import com.metawebthree.message.domain.model.Notification;
import com.metawebthree.common.constants.HeaderConstants;
import com.metawebthree.common.dto.ApiResponse;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/notification")
@RequiredArgsConstructor
@Tag(name = "Notification Controller", description = "消息通知接口")
public class NotificationController {

    private final NotificationService notificationService;

    @Operation(summary = "获取通知列表")
    @GetMapping("/list")
    public ApiResponse<List<Notification>> listNotifications(
            @RequestHeader(HeaderConstants.USER_ID) Long userId,
            @RequestParam(required = false) String type) {
        return ApiResponse.success(notificationService.listByUser(userId, type));
    }

    @Operation(summary = "标记已读")
    @PostMapping("/read")
    public ApiResponse<Void> markAsRead(@RequestHeader(HeaderConstants.USER_ID) Long userId,
            @RequestParam Long notificationId) {
        notificationService.markAsRead(userId, notificationId);
        return ApiResponse.success();
    }

    @Operation(summary = "全部标记已读")
    @PostMapping("/readAll")
    public ApiResponse<Void> markAllAsRead(@RequestHeader(HeaderConstants.USER_ID) Long userId) {
        notificationService.markAllAsRead(userId);
        return ApiResponse.success();
    }

    @Operation(summary = "删除通知")
    @DeleteMapping("/delete")
    public ApiResponse<Void> deleteNotification(@RequestHeader(HeaderConstants.USER_ID) Long userId,
            @RequestParam Long notificationId) {
        notificationService.deleteNotification(userId, notificationId);
        return ApiResponse.success();
    }

    @Operation(summary = "未读数量")
    @GetMapping("/unreadCount")
    public ApiResponse<Long> getUnreadCount(@RequestHeader(HeaderConstants.USER_ID) Long userId) {
        return ApiResponse.success(notificationService.getUnreadCount(userId));
    }

    @Operation(summary = "发送通知（内部调用）")
    @PostMapping("/send")
    public ApiResponse<Void> sendNotification(@RequestHeader(HeaderConstants.USER_ID) Long userId,
            @RequestBody NotificationParam param) {
        notificationService.createNotification(userId, param.getTitle(), param.getContent(),
                param.getType(), param.getRelatedId(), param.getIcon(), param.getImageUrl());
        return ApiResponse.success();
    }

    @Getter
    public static class NotificationParam {
        private String title;
        private String content;
        private String type;
        private String relatedId;
        private String icon;
        private String imageUrl;
    }
}
