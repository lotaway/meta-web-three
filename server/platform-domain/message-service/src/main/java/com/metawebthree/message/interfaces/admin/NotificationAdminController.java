package com.metawebthree.message.interfaces.admin;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.message.domain.model.Notification;
import com.metawebthree.message.infrastructure.persistence.mapper.NotificationMapper;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/admin/notification")
@RequiredArgsConstructor
public class NotificationAdminController {

    private final NotificationMapper notificationMapper;

    @GetMapping("/list")
    public ApiResponse<Map<String, Object>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) Long userId,
            @RequestParam(required = false) String title,
            @RequestParam(required = false) String type,
            @RequestParam(required = false) Integer readStatus,
            @RequestParam(required = false) String startDate,
            @RequestParam(required = false) String endDate) {

        LambdaQueryWrapper<Notification> query = new LambdaQueryWrapper<Notification>();

        if (userId != null) {
            query.eq(Notification::getUserId, userId);
        }
        if (title != null && !title.isEmpty()) {
            query.like(Notification::getTitle, title);
        }
        if (type != null && !type.isEmpty()) {
            query.eq(Notification::getType, type);
        }
        if (readStatus != null) {
            query.eq(Notification::getReadStatus, readStatus);
        }
        if (startDate != null && !startDate.isEmpty()) {
            query.ge(Notification::getCreateTime, LocalDateTime.parse(startDate + "T00:00:00"));
        }
        if (endDate != null && !endDate.isEmpty()) {
            query.le(Notification::getCreateTime, LocalDateTime.parse(endDate + "T23:59:59"));
        }

        query.orderByDesc(Notification::getCreateTime);

        Page<Notification> page = new Page<>(pageNum, pageSize);
        Page<Notification> result = notificationMapper.selectPage(page, query);

        Map<String, Object> response = new HashMap<>();
        response.put("list", result.getRecords());
        response.put("total", result.getTotal());
        response.put("pageNum", result.getCurrent());
        response.put("pageSize", result.getSize());

        return ApiResponse.success(response);
    }

    @GetMapping("/{id}")
    public ApiResponse<Notification> getById(@PathVariable Long id) {
        Notification notification = notificationMapper.selectById(id);
        if (notification != null) {
            return ApiResponse.success(notification);
        }
        return ApiResponse.error(ResponseStatus.NOT_FOUND, "Notification not found");
    }

    @PostMapping("/create")
    public ApiResponse<Notification> create(@Valid @RequestBody NotificationCreateParam param) {
        Notification notification = Notification.builder()
                .userId(param.getUserId())
                .title(param.getTitle())
                .content(param.getContent())
                .type(param.getType())
                .relatedId(param.getRelatedId())
                .icon(param.getIcon())
                .imageUrl(param.getImageUrl())
                .readStatus(0)
                .createTime(LocalDateTime.now())
                .build();
        notificationMapper.insert(notification);
        return ApiResponse.success(notification);
    }

    @PostMapping("/batch-create")
    public ApiResponse<Void> batchCreate(@Valid @RequestBody NotificationBatchCreateParam param) {
        if (param.getUserIds() == null || param.getUserIds().isEmpty()) {
            return ApiResponse.error(ResponseStatus.PARAM_ERROR, "userIds cannot be empty");
        }
        LocalDateTime now = LocalDateTime.now();
        for (Long userId : param.getUserIds()) {
            Notification notification = Notification.builder()
                    .userId(userId)
                    .title(param.getTitle())
                    .content(param.getContent())
                    .type(param.getType())
                    .relatedId(param.getRelatedId())
                    .icon(param.getIcon())
                    .imageUrl(param.getImageUrl())
                    .readStatus(0)
                    .createTime(now)
                    .build();
            notificationMapper.insert(notification);
        }
        return ApiResponse.success();
    }

    @DeleteMapping("/{id}")
    public ApiResponse<Void> delete(@PathVariable Long id) {
        int result = notificationMapper.deleteById(id);
        if (result > 0) {
            return ApiResponse.success();
        }
        return ApiResponse.error(ResponseStatus.NOT_FOUND, "Notification not found");
    }

    @DeleteMapping("/batch-delete")
    public ApiResponse<Void> batchDelete(@RequestBody List<Long> ids) {
        int count = 0;
        for (Long id : ids) {
            count += notificationMapper.deleteById(id);
        }
        return ApiResponse.success();
    }

    @GetMapping("/statistics")
    public ApiResponse<Map<String, Object>> getStatistics() {
        Long total = notificationMapper.selectCount(null);
        Long unread = notificationMapper.selectCount(new LambdaQueryWrapper<Notification>()
                .eq(Notification::getReadStatus, 0));
        Long read = notificationMapper.selectCount(new LambdaQueryWrapper<Notification>()
                .eq(Notification::getReadStatus, 1));

        Map<String, Object> statistics = new HashMap<>();
        statistics.put("total", total);
        statistics.put("unread", unread);
        statistics.put("read", read);

        return ApiResponse.success(statistics);
    }

    @GetMapping("/export")
    public ApiResponse<List<Notification>> export(
            @RequestParam(required = false) Long userId,
            @RequestParam(required = false) String title,
            @RequestParam(required = false) String type,
            @RequestParam(required = false) Integer readStatus) {

        LambdaQueryWrapper<Notification> query = new LambdaQueryWrapper<Notification>();

        if (userId != null) {
            query.eq(Notification::getUserId, userId);
        }
        if (title != null && !title.isEmpty()) {
            query.like(Notification::getTitle, title);
        }
        if (type != null && !type.isEmpty()) {
            query.eq(Notification::getType, type);
        }
        if (readStatus != null) {
            query.eq(Notification::getReadStatus, readStatus);
        }

        query.orderByDesc(Notification::getCreateTime);

        List<Notification> list = notificationMapper.selectList(query);
        return ApiResponse.success(list);
    }

    public static class NotificationCreateParam {
        @NotBlank(message = "title cannot be blank")
        private String title;
        @NotBlank(message = "content cannot be blank")
        private String content;
        private Long userId;
        private String type;
        private String relatedId;
        private String icon;
        private String imageUrl;

        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        public String getRelatedId() { return relatedId; }
        public void setRelatedId(String relatedId) { this.relatedId = relatedId; }
        public String getIcon() { return icon; }
        public void setIcon(String icon) { this.icon = icon; }
        public String getImageUrl() { return imageUrl; }
        public void setImageUrl(String imageUrl) { this.imageUrl = imageUrl; }
    }

    public static class NotificationBatchCreateParam {
        @NotEmpty(message = "userIds cannot be empty")
        private List<Long> userIds;
        @NotBlank(message = "title cannot be blank")
        private String title;
        @NotBlank(message = "content cannot be blank")
        private String content;
        private String type;
        private String relatedId;
        private String icon;
        private String imageUrl;

        public List<Long> getUserIds() { return userIds; }
        public void setUserIds(List<Long> userIds) { this.userIds = userIds; }
        public String getTitle() { return title; }
        public void setTitle(String title) { this.title = title; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        public String getRelatedId() { return relatedId; }
        public void setRelatedId(String relatedId) { this.relatedId = relatedId; }
        public String getIcon() { return icon; }
        public void setIcon(String icon) { this.icon = icon; }
        public String getImageUrl() { return imageUrl; }
        public void setImageUrl(String imageUrl) { this.imageUrl = imageUrl; }
    }
}