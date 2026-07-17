package com.metawebthree.reporting.interfaces.controller;

import com.metawebthree.reporting.domain.entity.ReportSubscription;
import com.metawebthree.reporting.domain.entity.ReportSubscription.Channel;
import com.metawebthree.reporting.domain.entity.ReportSubscription.Frequency;
import com.metawebthree.reporting.domain.entity.ReportSubscription.ReportType;
import com.metawebthree.reporting.domain.service.ReportSubscriptionService;
import jakarta.validation.Valid;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/report-subscription")
@RequiredArgsConstructor
public class ReportSubscriptionController {

    private final ReportSubscriptionService subscriptionService;

    @PostMapping("/email")
    public ResponseEntity<Long> createEmailSubscription(@Valid @RequestBody CreateEmailSubscriptionRequest request) {
        Long id = subscriptionService.createSubscription(
                request.getUserId(),
                request.getUserName(),
                request.getReportType(),
                request.getFrequency(),
                Channel.EMAIL,
                request.getRecipient()
        );
        return ResponseEntity.ok(id);
    }

    @PostMapping("/dingtalk")
    public ResponseEntity<Long> createDingTalkSubscription(@Valid @RequestBody CreateDingTalkSubscriptionRequest request) {
        Long id = subscriptionService.createDingTalkSubscription(
                request.getUserId(),
                request.getUserName(),
                request.getReportType(),
                request.getFrequency(),
                request.getWebhookUrl()
        );
        return ResponseEntity.ok(id);
    }

    @PutMapping("/{id}")
    public ResponseEntity<Void> updateSubscription(
            @PathVariable Long id,
            @Valid @RequestBody UpdateSubscriptionRequest request) {
        subscriptionService.updateSubscription(
                id,
                request.getFrequency(),
                request.getChannel(),
                request.getRecipient(),
                request.getEnabled()
        );
        return ResponseEntity.ok().build();
    }

    @PutMapping("/{id}/toggle")
    public ResponseEntity<Void> toggleSubscription(
            @PathVariable Long id,
            @RequestParam Boolean enabled) {
        subscriptionService.toggleSubscription(id, enabled);
        return ResponseEntity.ok().build();
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteSubscription(@PathVariable Long id) {
        subscriptionService.deleteSubscription(id);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/user/{userId}")
    public ResponseEntity<List<ReportSubscription>> getUserSubscriptions(@PathVariable Long userId) {
        List<ReportSubscription> subscriptions = subscriptionService.getUserSubscriptions(userId);
        return ResponseEntity.ok(subscriptions);
    }

    @GetMapping("/{id}")
    public ResponseEntity<ReportSubscription> getSubscription(@PathVariable Long id) {
        ReportSubscription subscription = subscriptionService.getSubscription(id);
        if (subscription != null) {
            return ResponseEntity.ok(subscription);
        }
        return ResponseEntity.notFound().build();
    }

    @Data
    public static class CreateEmailSubscriptionRequest {
        @NotNull
        private Long userId;
        @NotBlank
        private String userName;
        @NotNull
        private ReportType reportType;
        @NotNull
        private Frequency frequency;
        @NotBlank @Email
        private String recipient;
    }

    @Data
    public static class CreateDingTalkSubscriptionRequest {
        @NotNull
        private Long userId;
        @NotBlank
        private String userName;
        @NotNull
        private ReportType reportType;
        @NotNull
        private Frequency frequency;
        @NotBlank
        private String webhookUrl;
    }

    @Data
    public static class UpdateSubscriptionRequest {
        private Frequency frequency;
        private Channel channel;
        private String recipient;
        private Boolean enabled;
    }
}