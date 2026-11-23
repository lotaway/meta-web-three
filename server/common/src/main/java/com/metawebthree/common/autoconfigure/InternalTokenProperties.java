package com.metawebthree.common.autoconfigure;

import java.util.List;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Shared settings for the gateway-to-service trust token.
 *
 * <p>The gateway signs a short-lived HMAC token and puts it in
 * {@code X-Internal-Token}. Downstream services accept an incoming request only
 * when that token validates, which is what makes "no direct service calls"
 * enforceable instead of just a convention. The filter is opt-in per service
 * because a few services still receive internal HTTP calls that have not been
 * migrated to the gateway yet.</p>
 */
@ConfigurationProperties(prefix = "internal-token")
public class InternalTokenProperties {

    private boolean enabled = false;
    private String secret;
    private long timeoutSeconds = 5;
    private List<String> excludedPaths = List.of(
            "/actuator/**", "/health", "/info", "/error",
            "/swagger-ui/**", "/v3/api-docs/**");

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public String getSecret() {
        return secret;
    }

    public void setSecret(String secret) {
        this.secret = secret;
    }

    public long getTimeoutSeconds() {
        return timeoutSeconds;
    }

    public void setTimeoutSeconds(long timeoutSeconds) {
        this.timeoutSeconds = timeoutSeconds;
    }

    public List<String> getExcludedPaths() {
        return excludedPaths;
    }

    public void setExcludedPaths(List<String> excludedPaths) {
        this.excludedPaths = excludedPaths;
    }
}
