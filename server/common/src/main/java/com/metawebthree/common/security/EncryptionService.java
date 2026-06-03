package com.metawebthree.common.security;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import javax.crypto.Cipher;
import javax.crypto.SecretKey;
import javax.crypto.SecretKeyFactory;
import javax.crypto.spec.PBEKeySpec;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.security.spec.KeySpec;
import java.util.Base64;

/**
 * Encryption service for sensitive data storage
 * Supports AES encryption for field-level data encryption
 * Uses PBKDF2 for key derivation from password
 */
@Slf4j
@Service
public class EncryptionService {
    
    private static final String AES_ALGORITHM = "AES";
    private static final String AES_TRANSFORMATION = "AES/ECB/PKCS5Padding";
    private static final int AES_KEY_SIZE = 256;
    private static final int PBKDF2_ITERATIONS = 65536;
    private static final int PBKDF2_KEY_LENGTH = 256;
    private static final int SALT_LENGTH = 16;
    
    @Value("${security.encryption.password:metawebthree-encryption-password}")
    private String encryptionPassword;
    
    @Value("${security.encryption.salt:metawebthree-salt}")
    private String encryptionSalt;
    
    private SecretKey secretKey;
    
    /**
     * Initialize encryption service, generate AES key from password
     */
    @PostConstruct
    public void init() {
        try {
            // Use PBKDF2 to derive key from password
            SecretKeyFactory factory = SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256");
            KeySpec spec = new PBEKeySpec(encryptionPassword.toCharArray(), 
                                          encryptionSalt.getBytes(StandardCharsets.UTF_8), 
                                          PBKDF2_ITERATIONS, 
                                          PBKDF2_KEY_LENGTH);
            SecretKey tmp = factory.generateSecret(spec);
            secretKey = new SecretKeySpec(tmp.getEncoded(), AES_ALGORITHM);
            
            log.info("EncryptionService initialized successfully");
        } catch (Exception e) {
            log.error("Failed to initialize EncryptionService", e);
            throw new RuntimeException("Encryption initialization failed", e);
        }
    }
    
    /**
     * Encrypt plaintext
     * 
     * @param plaintext Plaintext to encrypt
     * @return Base64 encoded ciphertext
     */
    public String encrypt(String plaintext) {
        if (plaintext == null || plaintext.isEmpty()) {
            return plaintext;
        }
        
        try {
            Cipher cipher = Cipher.getInstance(AES_TRANSFORMATION);
            cipher.init(Cipher.ENCRYPT_MODE, secretKey);
            byte[] encryptedBytes = cipher.doFinal(plaintext.getBytes(StandardCharsets.UTF_8));
            return Base64.getEncoder().encodeToString(encryptedBytes);
        } catch (Exception e) {
            log.error("Encryption failed for plaintext", e);
            throw new RuntimeException("Encryption failed", e);
        }
    }
    
    /**
     * Decrypt ciphertext
     * 
     * @param ciphertext Base64 encoded ciphertext
     * @return Decrypted plaintext
     */
    public String decrypt(String ciphertext) {
        if (ciphertext == null || ciphertext.isEmpty()) {
            return ciphertext;
        }
        
        try {
            Cipher cipher = Cipher.getInstance(AES_TRANSFORMATION);
            cipher.init(Cipher.DECRYPT_MODE, secretKey);
            byte[] decodedBytes = Base64.getDecoder().decode(ciphertext);
            byte[] decryptedBytes = cipher.doFinal(decodedBytes);
            return new String(decryptedBytes, StandardCharsets.UTF_8);
        } catch (Exception e) {
            log.error("Decryption failed for ciphertext", e);
            throw new RuntimeException("Decryption failed", e);
        }
    }
    
    /**
     * Encrypt phone number
     * 
     * @param phone Phone number
     * @return Encrypted phone
     */
    public String encryptPhone(String phone) {
        if (phone == null || phone.isEmpty()) {
            return phone;
        }
        return encrypt(phone);
    }
    
    /**
     * Decrypt phone number
     * 
     * @param encryptedPhone Encrypted phone number
     * @return Decrypted phone number
     */
    public String decryptPhone(String encryptedPhone) {
        return decrypt(encryptedPhone);
    }
    
    /**
     * Encrypt ID card number
     * 
     * @param idCard ID card number
     * @return Encrypted ID card
     */
    public String encryptIdCard(String idCard) {
        if (idCard == null || idCard.isEmpty()) {
            return idCard;
        }
        return encrypt(idCard);
    }
    
    /**
     * Decrypt ID card number
     * 
     * @param encryptedIdCard Encrypted ID card number
     * @return Decrypted ID card number
     */
    public String decryptIdCard(String encryptedIdCard) {
        return decrypt(encryptedIdCard);
    }
    
    /**
     * Encrypt email address
     * 
     * @param email Email address
     * @return Encrypted email
     */
    public String encryptEmail(String email) {
        if (email == null || email.isEmpty()) {
            return email;
        }
        return encrypt(email);
    }
    
    /**
     * Decrypt email address
     * 
     * @param encryptedEmail Encrypted email
     * @return Decrypted email
     */
    public String decryptEmail(String encryptedEmail) {
        return decrypt(encryptedEmail);
    }
    
    /**
     * Generate random salt for additional security (for future use)
     * 
     * @return Random salt bytes
     */
    private byte[] generateSalt() {
        java.security.SecureRandom random = new java.security.SecureRandom();
        byte[] salt = new byte[SALT_LENGTH];
        random.nextBytes(salt);
        return salt;
    }
}
