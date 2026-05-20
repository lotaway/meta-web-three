package com.metawebthree.user.application;

import java.util.List;
import java.util.Map;

public interface PasskeyService {
    Map<String, Object> generateRegistrationOptions(Long userId, String rpId);
    Long verifyAndStoreCredential(Long userId, String rpId, Map<String, Object> attestation);
    Map<String, Object> generateAuthenticationOptions(String rpId);
    Long verifyAndAuthenticate(String rpId, Map<String, Object> assertion);
    List<Map<String, Object>> getPasskeyList(Long userId);
    boolean deletePasskey(Long userId, String credentialId);
}
