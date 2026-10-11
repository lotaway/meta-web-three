package com.metawebthree.developerportal.service;

import com.metawebthree.developerportal.dto.*;
import com.metawebthree.developerportal.entity.ApiDeveloper;
import com.metawebthree.developerportal.entity.OAuthApplication;
import com.metawebthree.developerportal.entity.OAuthApplication.AppType;
import com.metawebthree.developerportal.repository.ApiDeveloperRepository;
import com.metawebthree.developerportal.repository.OAuthApplicationRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class OAuthApplicationService {

    private final OAuthApplicationRepository oauthRepository;
    private final ApiDeveloperRepository developerRepository;
    private final PasswordEncoder passwordEncoder;

    @Transactional
    public OAuthAppResponse registerApplication(String developerId, OAuthAppCreateRequest request) {
        ApiDeveloper developer = developerRepository.findByDeveloperId(developerId)
            .orElseThrow(() -> new IllegalArgumentException("Developer not found: " + developerId));

        if (developer.getStatus() != ApiDeveloper.DeveloperStatus.APPROVED) {
            throw new IllegalStateException("Developer is not approved: " + developer.getStatus());
        }

        String clientId = generateClientId();
        String clientSecret = generateClientSecret();

        OAuthApplication app = new OAuthApplication();
        app.setClientId(clientId);
        app.setClientSecret(passwordEncoder.encode(clientSecret));
        app.setDeveloperId(developerId);
        app.setName(request.getName());
        app.setDescription(request.getDescription());
        app.setRedirectUris(request.getRedirectUris());
        app.setAppType(AppType.valueOf(request.getAppType().toUpperCase()));
        app.setGrantTypes(request.getGrantTypes());
        app.setScopes(request.getScopes());
        app.setStatus(OAuthApplication.AppStatus.ACTIVE);

        oauthRepository.save(app);
        log.info("OAuth Application registered: {} for developer {}", clientId, developerId);

        OAuthAppResponse response = OAuthAppResponse.fromEntity(app);
        response.setClientSecret(clientSecret);
        return response;
    }

    public boolean validateClient(String clientId, String clientSecret) {
        OAuthApplication app = oauthRepository.findByClientId(clientId).orElse(null);

        if (app == null) {
            log.warn("OAuth client not found: {}", clientId);
            return false;
        }

        if (app.getStatus() != OAuthApplication.AppStatus.ACTIVE) {
            log.warn("OAuth client is not active: {} - {}", clientId, app.getStatus());
            return false;
        }

        if (app.getAppType() == AppType.PUBLIC) {
            return true;
        }

        if (!passwordEncoder.matches(clientSecret, app.getClientSecret())) {
            log.warn("OAuth client secret mismatch: {}", clientId);
            return false;
        }

        return true;
    }

    public OAuthAppResponse getApplication(String clientId) {
        OAuthApplication app = oauthRepository.findByClientId(clientId)
            .orElseThrow(() -> new IllegalArgumentException("OAuth application not found: " + clientId));
        return OAuthAppResponse.fromEntity(app);
    }

    public List<OAuthAppResponse> getDeveloperApplications(String developerId) {
        return oauthRepository.findByDeveloperId(developerId).stream()
            .map(OAuthAppResponse::fromEntity)
            .collect(Collectors.toList());
    }

    @Transactional
    public OAuthAppResponse updateApplication(String clientId, OAuthAppCreateRequest request) {
        OAuthApplication app = oauthRepository.findByClientId(clientId)
            .orElseThrow(() -> new IllegalArgumentException("OAuth application not found: " + clientId));

        app.setName(request.getName());
        app.setDescription(request.getDescription());
        app.setRedirectUris(request.getRedirectUris());
        app.setGrantTypes(request.getGrantTypes());
        app.setScopes(request.getScopes());

        oauthRepository.save(app);
        log.info("OAuth Application updated: {}", clientId);

        return OAuthAppResponse.fromEntity(app);
    }

    @Transactional
    public OAuthAppResponse regenerateClientSecret(String clientId) {
        OAuthApplication app = oauthRepository.findByClientId(clientId)
            .orElseThrow(() -> new IllegalArgumentException("OAuth application not found: " + clientId));

        String newSecret = generateClientSecret();
        app.setClientSecret(passwordEncoder.encode(newSecret));
        oauthRepository.save(app);
        log.info("OAuth client secret regenerated: {}", clientId);

        OAuthAppResponse response = OAuthAppResponse.fromEntity(app);
        response.setClientSecret(newSecret);
        return response;
    }

    @Transactional
    public OAuthAppResponse disableApplication(String clientId) {
        OAuthApplication app = oauthRepository.findByClientId(clientId)
            .orElseThrow(() -> new IllegalArgumentException("OAuth application not found: " + clientId));

        app.setStatus(OAuthApplication.AppStatus.DISABLED);
        oauthRepository.save(app);
        log.info("OAuth Application disabled: {}", clientId);

        return OAuthAppResponse.fromEntity(app);
    }

    @Transactional
    public OAuthAppResponse enableApplication(String clientId) {
        OAuthApplication app = oauthRepository.findByClientId(clientId)
            .orElseThrow(() -> new IllegalArgumentException("OAuth application not found: " + clientId));

        app.setStatus(OAuthApplication.AppStatus.ACTIVE);
        oauthRepository.save(app);
        log.info("OAuth Application enabled: {}", clientId);

        return OAuthAppResponse.fromEntity(app);
    }

    @Transactional
    public void deleteApplication(String clientId) {
        OAuthApplication app = oauthRepository.findByClientId(clientId)
            .orElseThrow(() -> new IllegalArgumentException("OAuth application not found: " + clientId));

        oauthRepository.delete(app);
        log.info("OAuth Application deleted: {}", clientId);
    }

    public boolean isValidRedirectUri(String clientId, String redirectUri) {
        OAuthApplication app = oauthRepository.findByClientId(clientId)
            .orElse(null);

        if (app == null || app.getRedirectUris() == null) {
            return false;
        }

        String[] allowedUris = app.getRedirectUris().split(",");
        for (String allowed : allowedUris) {
            if (allowed.trim().equals(redirectUri)) {
                return true;
            }
        }

        return false;
    }

    public boolean isGrantTypeAllowed(String clientId, String grantType) {
        OAuthApplication app = oauthRepository.findByClientId(clientId)
            .orElse(null);

        if (app == null || app.getGrantTypes() == null) {
            return false;
        }

        String[] allowedTypes = app.getGrantTypes().split(",");
        for (String allowed : allowedTypes) {
            if (allowed.trim().equals(grantType)) {
                return true;
            }
        }

        return false;
    }

    private String generateClientId() {
        return "oauth_" + UUID.randomUUID().toString().replace("-", "").substring(0, 20);
    }

    private String generateClientSecret() {
        return UUID.randomUUID().toString().replace("-", "") + 
               UUID.randomUUID().toString().replace("-", "").substring(0, 16);
    }
}
