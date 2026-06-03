package com.metawebthree.common.security;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.TestPropertySource;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for EncryptionService
 * Verifies AES encryption/decryption functionality
 */
@SpringBootTest
@TestPropertySource(properties = {
    "security.encryption.password=test-encryption-password",
    "security.encryption.salt=test-encryption-salt"
})
public class EncryptionServiceTest {
    
    @Autowired
    private EncryptionService encryptionService;
    
    @Test
    public void testEncryptDecryptPhone() {
        // Given
        String originalPhone = "13800138000";
        
        // When
        String encrypted = encryptionService.encryptPhone(originalPhone);
        String decrypted = encryptionService.decryptPhone(encrypted);
        
        // Then
        assertNotNull(encrypted);
        assertNotNull(decrypted);
        assertEquals(originalPhone, decrypted);
        assertNotEquals(originalPhone, encrypted); // Encrypted should be different from original
    }
    
    @Test
    public void testEncryptDecryptEmail() {
        // Given
        String originalEmail = "test@example.com";
        
        // When
        String encrypted = encryptionService.encryptEmail(originalEmail);
        String decrypted = encryptionService.decryptEmail(encrypted);
        
        // Then
        assertNotNull(encrypted);
        assertNotNull(decrypted);
        assertEquals(originalEmail, decrypted);
        assertNotEquals(originalEmail, encrypted);
    }
    
    @Test
    public void testEncryptDecryptIdCard() {
        // Given
        String originalIdCard = "110101199003078274";
        
        // When
        String encrypted = encryptionService.encryptIdCard(originalIdCard);
        String decrypted = encryptionService.decryptIdCard(encrypted);
        
        // Then
        assertNotNull(encrypted);
        assertNotNull(decrypted);
        assertEquals(originalIdCard, decrypted);
        assertNotEquals(originalIdCard, encrypted);
    }
    
    @Test
    public void testEncryptDecryptGeneric() {
        // Given
        String plaintext = "Sensitive Data 123!@#";
        
        // When
        String encrypted = encryptionService.encrypt(plaintext);
        String decrypted = encryptionService.decrypt(encrypted);
        
        // Then
        assertNotNull(encrypted);
        assertNotNull(decrypted);
        assertEquals(plaintext, decrypted);
        assertNotEquals(plaintext, encrypted);
    }
    
    @Test
    public void testEncryptNullAndEmpty() {
        // Test null
        assertNull(encryptionService.encrypt(null));
        assertNull(encryptionService.decrypt(null));
        
        // Test empty string
        assertEquals("", encryptionService.encrypt(""));
        assertEquals("", encryptionService.decrypt(""));
    }
    
    @Test
    public void testEncryptionConsistency() {
        // Given
        String plaintext = "TestConsistency";
        
        // When
        String encrypted1 = encryptionService.encrypt(plaintext);
        String encrypted2 = encryptionService.encrypt(plaintext);
        
        // Then - same plaintext should produce same ciphertext (deterministic)
        assertEquals(encrypted1, encrypted2);
    }
}