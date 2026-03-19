package com.metawebthree.user.infrastructure.client;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import com.metawebthree.user.domain.ports.ReferralBindingPort;

@Component
public class CommissionClient implements ReferralBindingPort {
    private final RestTemplate restTemplate;
    private final String baseUrl;

    public CommissionClient(RestTemplateBuilder builder,
            @Value("${commission.service.base-url}") String baseUrl) {
        this.restTemplate = builder.build();
        this.baseUrl = baseUrl;
    }

    @Override
    public void bind(Long userId, Long referrerId) {
        validate(userId, referrerId);
        BindRequest request = new BindRequest(userId, referrerId);
        restTemplate.postForEntity(baseUrl + "/v1/commission/relations/bind", request, Void.class);
    }

    private void validate(Long userId, Long referrerId) {
        if (userId == null || referrerId == null || referrerId <= 0) {
            throw new IllegalArgumentException("invalid referral binding");
        }
    }

    private static class BindRequest {
        private Long userId;
        private Long parentUserId;

        BindRequest(Long userId, Long referrerId) {
            this.userId = userId;
            this.parentUserId = referrerId;
        }

        public Long getUserId() { return userId; }
        public void setUserId(Long userId) { this.userId = userId; }
        public Long getParentUserId() { return parentUserId; }
        public void setParentUserId(Long parentUserId) { this.parentUserId = parentUserId; }
    }
}
