package com.metawebthree.user.application;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.user.domain.model.PasskeyCredentialDO;
import com.metawebthree.user.infrastructure.persistence.mapper.PasskeyCredentialMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.*;

@Slf4j
@Service
public class PasskeyServiceImpl implements PasskeyService {

    private final PasskeyCredentialMapper passkeyCredentialMapper;

    public PasskeyServiceImpl(PasskeyCredentialMapper passkeyCredentialMapper) {
        this.passkeyCredentialMapper = passkeyCredentialMapper;
    }

    @Override
    public Map<String, Object> generateRegistrationOptions(Long userId, String rpId) {
        String challenge = generateSecureChallenge();
        String challengeBase64 = Base64.getUrlEncoder().withoutPadding().encodeToString(challenge.getBytes(StandardCharsets.UTF_8));

        Map<String, Object> options = new HashMap<>();
        options.put("challenge", challengeBase64);
        options.put("rp", Map.of("id", rpId, "name", "Meta Web Three"));
        options.put("pubKeyCredParams", List.of(Map.of("type", "public-key", "alg", -7)));
        options.put("timeout", 60000);
        options.put("authenticatorSelection", Map.of(
            "authenticatorAttachment", "platform",
            "residentKey", "required",
            "userVerification", "required"
        ));
        options.put("attestation", "none");

        return options;
    }

    @Override
    @Transactional
    public Long verifyAndStoreCredential(Long userId, String rpId, Map<String, Object> attestation) {
        Map<String, Object> response = (Map<String, Object>) attestation.get("response");
        String credentialId = (String) attestation.get("id");
        String rawId = (String) attestation.get("rawId");

        String storedCredentialId = rawId != null ? rawId : credentialId;
        String publicKey = encodePublicKey(response);

        PasskeyCredentialDO credential = PasskeyCredentialDO.builder()
            .id(IdWorker.getId())
            .userId(userId)
            .credentialId(storedCredentialId)
            .publicKey(publicKey)
            .rpId(rpId)
            .counter(0L)
            .deviceType("platform")
            .createdAt(new Date())
            .isRevoked(false)
            .build();

        passkeyCredentialMapper.insert(credential);
        log.info("Passkey credential stored for userId: {}", userId);
        return userId;
    }

    @Override
    public Map<String, Object> generateAuthenticationOptions(String rpId) {
        String challenge = generateSecureChallenge();
        String challengeBase64 = Base64.getUrlEncoder().withoutPadding().encodeToString(challenge.getBytes(StandardCharsets.UTF_8));

        List<PasskeyCredentialDO> credentials = passkeyCredentialMapper.selectByUserId(null);
        List<String> allowedCredentials = new ArrayList<>();
        for (PasskeyCredentialDO cred : credentials) {
            if (!cred.getIsRevoked() && rpId.equals(cred.getRpId())) {
                allowedCredentials.add(cred.getCredentialId());
            }
        }

        Map<String, Object> options = new HashMap<>();
        options.put("challenge", challengeBase64);
        options.put("rpId", rpId);
        options.put("userVerification", "required");
        options.put("timeout", 60000);

        if (!allowedCredentials.isEmpty()) {
            options.put("allowCredentials", allowedCredentials.stream()
                .map(id -> Map.of("type", "public-key", "id", id))
                .toList());
        }

        return options;
    }

    @Override
    public Long verifyAndAuthenticate(String rpId, Map<String, Object> assertion) {
        Map<String, Object> response = (Map<String, Object>) assertion.get("response");
        String credentialId = (String) assertion.get("id");

        PasskeyCredentialDO credential = passkeyCredentialMapper.selectByCredentialId(credentialId);
        if (credential == null || credential.getIsRevoked()) {
            throw new IllegalArgumentException("Invalid credential");
        }

        if (!rpId.equals(credential.getRpId())) {
            throw new IllegalArgumentException("RP ID mismatch");
        }

        credential.setCounter(credential.getCounter() + 1);
        credential.setLastUsedAt(new Date());
        passkeyCredentialMapper.updateById(credential);

        log.info("Passkey authenticated for userId: {}", credential.getUserId());
        return credential.getUserId();
    }

    @Override
    public List<Map<String, Object>> getPasskeyList(Long userId) {
        List<PasskeyCredentialDO> credentials = passkeyCredentialMapper.selectByUserId(userId);
        return credentials.stream()
            .filter(c -> !c.getIsRevoked())
            .map(c -> {
                Map<String, Object> item = new HashMap<>();
                item.put("credentialId", c.getCredentialId());
                item.put("rpId", c.getRpId());
                item.put("createdAt", c.getCreatedAt());
                item.put("lastUsedAt", c.getLastUsedAt());
                item.put("deviceType", c.getDeviceType());
                return item;
            })
            .toList();
    }

    @Override
    @Transactional
    public boolean deletePasskey(Long userId, String credentialId) {
        PasskeyCredentialDO credential = passkeyCredentialMapper.selectByCredentialId(credentialId);
        if (credential == null || !credential.getUserId().equals(userId)) {
            return false;
        }

        credential.setIsRevoked(true);
        passkeyCredentialMapper.updateById(credential);
        log.info("Passkey credential revoked for userId: {}, credentialId: {}", userId, credentialId);
        return true;
    }

    private String generateSecureChallenge() {
        byte[] bytes = new byte[32];
        new Random().nextBytes(bytes);
        return Base64.getUrlEncoder().withoutPadding().encodeToString(bytes);
    }

    private String encodePublicKey(Map<String, Object> response) {
        return "";
    }
}
