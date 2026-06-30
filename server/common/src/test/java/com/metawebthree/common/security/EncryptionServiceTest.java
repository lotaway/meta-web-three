package com.metawebthree.common.security;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.TestPropertySource;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
@ContextConfiguration(classes = EncryptionServiceTest.TestConfig.class)
@TestPropertySource(properties = {
    "security.encryption.password=test-encryption-password",
    "security.encryption.salt=test-encryption-salt"
})
public class EncryptionServiceTest {

    @Autowired
    private EncryptionService encryptionService;

    @Configuration
    static class TestConfig {
        @Bean
        public EncryptionService encryptionService() {
            return new EncryptionService();
        }
    }

    @Test
    public void testEncryptDecryptPhone() {
        String originalPhone = "13800138000";
        String encrypted = encryptionService.encryptPhone(originalPhone);
        String decrypted = encryptionService.decryptPhone(encrypted);
        assertNotNull(encrypted);
        assertNotNull(decrypted);
        assertEquals(originalPhone, decrypted);
        assertNotEquals(originalPhone, encrypted);
    }

    @Test
    public void testEncryptDecryptEmail() {
        String originalEmail = "test@example.com";
        String encrypted = encryptionService.encryptEmail(originalEmail);
        String decrypted = encryptionService.decryptEmail(encrypted);
        assertNotNull(encrypted);
        assertNotNull(decrypted);
        assertEquals(originalEmail, decrypted);
        assertNotEquals(originalEmail, encrypted);
    }

    @Test
    public void testEncryptDecryptIdCard() {
        String originalIdCard = "110101199003078274";
        String encrypted = encryptionService.encryptIdCard(originalIdCard);
        String decrypted = encryptionService.decryptIdCard(encrypted);
        assertNotNull(encrypted);
        assertNotNull(decrypted);
        assertEquals(originalIdCard, decrypted);
        assertNotEquals(originalIdCard, encrypted);
    }

    @Test
    public void testEncryptDecryptGeneric() {
        String plaintext = "Sensitive Data 123!@#";
        String encrypted = encryptionService.encrypt(plaintext);
        String decrypted = encryptionService.decrypt(encrypted);
        assertNotNull(encrypted);
        assertNotNull(decrypted);
        assertEquals(plaintext, decrypted);
        assertNotEquals(plaintext, encrypted);
    }

    @Test
    public void testEncryptNullAndEmpty() {
        assertNull(encryptionService.encrypt(null));
        assertNull(encryptionService.decrypt(null));
        assertEquals("", encryptionService.encrypt(""));
        assertEquals("", encryptionService.decrypt(""));
    }

    @Test
    public void testEncryptionConsistency() {
        String plaintext = "TestConsistency";
        String encrypted1 = encryptionService.encrypt(plaintext);
        String encrypted2 = encryptionService.encrypt(plaintext);
        assertEquals(encrypted1, encrypted2);
    }
}
