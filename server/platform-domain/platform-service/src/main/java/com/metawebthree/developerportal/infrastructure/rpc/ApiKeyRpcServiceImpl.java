package com.metawebthree.developerportal.infrastructure.rpc;

import java.time.LocalDateTime;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;

import org.apache.dubbo.config.annotation.DubboService;

import com.metawebthree.common.generated.rpc.platform.ApiKeyService;
import com.metawebthree.common.generated.rpc.platform.GetApiKeyRequest;
import com.metawebthree.common.generated.rpc.platform.GetApiKeyResponse;
import com.metawebthree.common.generated.rpc.platform.ValidateApiKeyRequest;
import com.metawebthree.common.generated.rpc.platform.ValidateApiKeyResponse;
import com.metawebthree.developerportal.entity.ApiKey;
import com.metawebthree.developerportal.repository.ApiKeyRepository;

/**
 * Dubbo provider for gateway API-key checks. Mirrors the validation rules of
 * {@code ApiKeyValidationController} so the gateway filter no longer has to
 * call platform-service over raw HTTP.
 */
@DubboService
public class ApiKeyRpcServiceImpl implements ApiKeyService {

    private final ApiKeyRepository apiKeyRepository;

    public ApiKeyRpcServiceImpl(ApiKeyRepository apiKeyRepository) {
        this.apiKeyRepository = apiKeyRepository;
    }

    @Override
    public ValidateApiKeyResponse validateApiKey(ValidateApiKeyRequest request) {
        Optional<ApiKey> found = apiKeyRepository.findByKeyId(request.getKeyId());
        if (found.isEmpty()) {
            return ValidateApiKeyResponse.newBuilder()
                    .setValid(false).setMessage("API key not found").build();
        }

        ApiKey apiKey = found.get();
        String rejection = rejectionReason(apiKey, request.getApiSecret());
        if (rejection != null) {
            return ValidateApiKeyResponse.newBuilder()
                    .setValid(false).setMessage(rejection).build();
        }

        apiKey.setLastUsedAt(LocalDateTime.now());
        apiKeyRepository.save(apiKey);

        ValidateApiKeyResponse.Builder response = ValidateApiKeyResponse.newBuilder()
                .setValid(true)
                .setKeyId(apiKey.getKeyId());
        if (apiKey.getDeveloperId() != null) {
            response.setDeveloperId(apiKey.getDeveloperId());
        }
        if (apiKey.getScopes() != null) {
            response.setScopes(apiKey.getScopes());
        }
        if (apiKey.getRateLimit() != null) {
            response.setRateLimit(apiKey.getRateLimit());
        }
        return response.build();
    }

    @Override
    public CompletableFuture<ValidateApiKeyResponse> validateApiKeyAsync(ValidateApiKeyRequest request) {
        return CompletableFuture.completedFuture(validateApiKey(request));
    }

    @Override
    public GetApiKeyResponse getApiKey(GetApiKeyRequest request) {
        Optional<ApiKey> found = apiKeyRepository.findByKeyId(request.getKeyId());
        if (found.isEmpty()) {
            return GetApiKeyResponse.newBuilder().setFound(false).build();
        }
        ApiKey apiKey = found.get();
        GetApiKeyResponse.Builder response = GetApiKeyResponse.newBuilder()
                .setFound(true)
                .setKeyId(apiKey.getKeyId());
        if (apiKey.getDeveloperId() != null) {
            response.setDeveloperId(apiKey.getDeveloperId());
        }
        return response.build();
    }

    @Override
    public CompletableFuture<GetApiKeyResponse> getApiKeyAsync(GetApiKeyRequest request) {
        return CompletableFuture.completedFuture(getApiKey(request));
    }

    private String rejectionReason(ApiKey apiKey, String apiSecret) {
        if (apiKey.getStatus() != ApiKey.KeyStatus.ACTIVE) {
            return "API key is " + apiKey.getStatus().name().toLowerCase();
        }
        if (apiKey.getExpiresAt() != null && apiKey.getExpiresAt().isBefore(LocalDateTime.now())) {
            return "API key has expired";
        }
        if (apiSecret == null || apiSecret.isEmpty()) {
            return "API secret is required";
        }
        return null;
    }
}
