package com.metawebthree.developerportal.controller;

import com.metawebthree.developerportal.entity.ApiDeveloper;
import com.metawebthree.developerportal.repository.ApiDeveloperRepository;
import com.metawebthree.developerportal.service.ApiBillingService;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/developer/billing")
@RequiredArgsConstructor
@Validated
public class ApiBillingController {

    private final ApiBillingService billingService;
    private final ApiDeveloperRepository developerRepository;

    @GetMapping("/summary")
    public ResponseEntity<?> getBillingSummary(@RequestParam String developerId) {
        try {
            ApiBillingService.BillingSummary summary = billingService.getBillingSummary(developerId);
            return ResponseEntity.ok(summary);
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().body(Map.of("error", e.getMessage()));
        }
    }

    @GetMapping("/quota-check")
    public ResponseEntity<?> checkQuota(
            @RequestParam String developerId,
            @RequestParam int dailyQuota,
            @RequestParam int monthlyQuota) {
        try {
            boolean exceeded = billingService.hasExceededQuota(developerId, dailyQuota, monthlyQuota);
            return ResponseEntity.ok(Map.of(
                "developerId", developerId,
                "quotaExceeded", exceeded
            ));
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().body(Map.of("error", e.getMessage()));
        }
    }

    @GetMapping("/usage")
    public ResponseEntity<?> getUsageStats(
            @RequestParam String developerId,
            @RequestParam String startTime,
            @RequestParam String endTime) {
        try {
            LocalDateTime start = LocalDateTime.parse(startTime, DateTimeFormatter.ISO_DATE_TIME);
            LocalDateTime end = LocalDateTime.parse(endTime, DateTimeFormatter.ISO_DATE_TIME);

            var stats = billingService.getUsageStats(developerId, start, end);
            return ResponseEntity.ok(stats);
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of("error", "Invalid time format: " + e.getMessage()));
        }
    }

    @GetMapping("/calculate")
    public ResponseEntity<?> calculateBilling(
            @RequestParam String developerId,
            @RequestParam String apiEndpoint,
            @RequestParam int responseTimeMs,
            @RequestParam int dataTransferredBytes) {
        try {
            long amount = billingService.calculateBillingAmount(
                developerId, apiEndpoint, responseTimeMs, dataTransferredBytes);

            return ResponseEntity.ok(Map.of(
                "developerId", developerId,
                "apiEndpoint", apiEndpoint,
                "responseTimeMs", responseTimeMs,
                "dataTransferredBytes", dataTransferredBytes,
                "billingAmountCents", amount,
                "billingAmountDollars", amount / 100.0
            ));
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().body(Map.of("error", e.getMessage()));
        }
    }

    @GetMapping("/admin/low-balance")
    public ResponseEntity<?> getDevelopersWithLowBalance(
            @RequestParam @Min(value = 0, message = "Threshold must be non-negative") long thresholdCents) {
        List<ApiDeveloper> developers = developerRepository.findByBalanceBelowThreshold(thresholdCents);

        List<Map<String, Object>> result = developers.stream()
            .map(d -> Map.<String, Object>of(
                "developerId", d.getDeveloperId(),
                "name", d.getName(),
                "email", d.getEmail(),
                "balanceCents", d.getBalance(),
                "billingPlan", d.getBillingPlan().name()
            ))
            .toList();

        log.info("Found {} developers with balance below {} cents", result.size(), thresholdCents);
        return ResponseEntity.ok(Map.of(
            "thresholdCents", thresholdCents,
            "count", result.size(),
            "developers", result
        ));
    }

    @PostMapping("/admin/topup")
    public ResponseEntity<?> topUpBalance(
            @RequestParam @NotBlank(message = "Developer ID is required") String developerId,
            @RequestParam @Min(value = 1, message = "Amount must be positive") long amountCents) {
        ApiDeveloper developer = developerRepository.findByDeveloperId(developerId)
            .orElseThrow(() -> new IllegalArgumentException("Developer not found: " + developerId));

        long previousBalance = developer.getBalance();
        long newBalance = previousBalance + amountCents;
        developer.setBalance(newBalance);
        developerRepository.save(developer);

        log.info("Topped up balance for developer {}: {} + {} = {} cents", 
            developerId, previousBalance, amountCents, newBalance);

        return ResponseEntity.ok(Map.of(
            "developerId", developerId,
            "previousBalanceCents", previousBalance,
            "topUpAmountCents", amountCents,
            "newBalanceCents", newBalance,
            "newBalanceDollars", newBalance / 100.0
        ));
    }
}
