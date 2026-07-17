package com.metawebthree.developerportal.service;

import com.metawebthree.developerportal.dto.*;
import com.metawebthree.developerportal.entity.ApiDeveloper;
import com.metawebthree.developerportal.entity.ApiDeveloper.DeveloperStatus;
import com.metawebthree.developerportal.repository.ApiDeveloperRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class ApiDeveloperService {

    private final ApiDeveloperRepository developerRepository;

    @Transactional
    public DeveloperResponse register(DeveloperRegistrationRequest request) {
        if (developerRepository.existsByEmail(request.getEmail())) {
            throw new IllegalArgumentException("Email already registered: " + request.getEmail());
        }

        ApiDeveloper developer = new ApiDeveloper();
        developer.setDeveloperId(generateDeveloperId());
        developer.setEmail(request.getEmail());
        developer.setName(request.getName());
        developer.setPhone(request.getPhone());
        developer.setDescription(request.getDescription());
        developer.setStatus(DeveloperStatus.PENDING);

        ApiDeveloper.BillingPlan freePlan = ApiDeveloper.BillingPlan.FREE;
        developer.setDailyQuota(freePlan.getDailyQuota());
        developer.setMonthlyQuota(freePlan.getMonthlyQuota());
        developer.setBillingPlan(freePlan);

        developerRepository.save(developer);
        log.info("Developer registered: {} ({})", developer.getName(), developer.getDeveloperId());

        return DeveloperResponse.fromEntity(developer);
    }

    @Transactional
    public DeveloperResponse approve(String developerId, String reviewedBy, String note) {
        ApiDeveloper developer = developerRepository.findByDeveloperId(developerId)
            .orElseThrow(() -> new IllegalArgumentException("Developer not found: " + developerId));

        if (developer.getStatus() != DeveloperStatus.PENDING) {
            throw new IllegalStateException("Developer is not in PENDING status: " + developer.getStatus());
        }

        developer.setStatus(DeveloperStatus.APPROVED);
        developer.setReviewedBy(reviewedBy);
        developer.setReviewNote(note);
        developer.setReviewedAt(LocalDateTime.now());

        developerRepository.save(developer);
        log.info("Developer approved: {} by {}", developerId, reviewedBy);

        return DeveloperResponse.fromEntity(developer);
    }

    @Transactional
    public DeveloperResponse reject(String developerId, String reviewedBy, String reason) {
        ApiDeveloper developer = developerRepository.findByDeveloperId(developerId)
            .orElseThrow(() -> new IllegalArgumentException("Developer not found: " + developerId));

        if (developer.getStatus() != DeveloperStatus.PENDING) {
            throw new IllegalStateException("Developer is not in PENDING status: " + developer.getStatus());
        }

        developer.setStatus(DeveloperStatus.REJECTED);
        developer.setReviewedBy(reviewedBy);
        developer.setReviewNote(reason);
        developer.setReviewedAt(LocalDateTime.now());

        developerRepository.save(developer);
        log.info("Developer rejected: {} by {}", developerId, reviewedBy);

        return DeveloperResponse.fromEntity(developer);
    }

    @Transactional
    public DeveloperResponse suspend(String developerId, String reason) {
        ApiDeveloper developer = developerRepository.findByDeveloperId(developerId)
            .orElseThrow(() -> new IllegalArgumentException("Developer not found: " + developerId));

        developer.setStatus(DeveloperStatus.SUSPENDED);
        developer.setReviewNote(reason);
        developer.setReviewedAt(LocalDateTime.now());

        developerRepository.save(developer);
        log.info("Developer suspended: {} - {}", developerId, reason);

        return DeveloperResponse.fromEntity(developer);
    }

    @Transactional
    public DeveloperResponse reactivate(String developerId) {
        ApiDeveloper developer = developerRepository.findByDeveloperId(developerId)
            .orElseThrow(() -> new IllegalArgumentException("Developer not found: " + developerId));

        developer.setStatus(DeveloperStatus.APPROVED);
        developer.setReviewedAt(LocalDateTime.now());

        developerRepository.save(developer);
        log.info("Developer reactivated: {}", developerId);

        return DeveloperResponse.fromEntity(developer);
    }

    @Transactional
    public DeveloperResponse updateBillingPlan(String developerId, ApiDeveloper.BillingPlan plan) {
        ApiDeveloper developer = developerRepository.findByDeveloperId(developerId)
            .orElseThrow(() -> new IllegalArgumentException("Developer not found: " + developerId));

        developer.setBillingPlan(plan);
        developer.setDailyQuota(plan.getDailyQuota());
        developer.setMonthlyQuota(plan.getMonthlyQuota());

        developerRepository.save(developer);
        log.info("Developer billing plan updated: {} -> {}", developerId, plan);

        return DeveloperResponse.fromEntity(developer);
    }

    public DeveloperResponse getDeveloper(String developerId) {
        ApiDeveloper developer = developerRepository.findByDeveloperId(developerId)
            .orElseThrow(() -> new IllegalArgumentException("Developer not found: " + developerId));
        return DeveloperResponse.fromEntity(developer);
    }

    public DeveloperResponse getDeveloperByEmail(String email) {
        ApiDeveloper developer = developerRepository.findByEmail(email)
            .orElseThrow(() -> new IllegalArgumentException("Developer not found for email: " + email));
        return DeveloperResponse.fromEntity(developer);
    }

    public List<DeveloperResponse> getPendingDevelopers() {
        return developerRepository.findByStatus(DeveloperStatus.PENDING).stream()
            .map(DeveloperResponse::fromEntity)
            .collect(Collectors.toList());
    }

    public List<DeveloperResponse> getApprovedDevelopers() {
        return developerRepository.findByStatus(DeveloperStatus.APPROVED).stream()
            .map(DeveloperResponse::fromEntity)
            .collect(Collectors.toList());
    }

    private String generateDeveloperId() {
        return "dev_" + UUID.randomUUID().toString().replace("-", "").substring(0, 16);
    }
}
