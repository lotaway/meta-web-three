package com.metawebthree.common.security;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Annotation to mark fields that need encryption/decryption
 * Used with EncryptionAspect to automatically encrypt/decrypt sensitive data
 */
@Target(ElementType.FIELD)
@Retention(RetentionPolicy.RUNTIME)
public @interface EncryptField {
    /**
     * Encryption type
     */
    EncryptionType value() default EncryptionType.AES;
    
    /**
     * Whether to encrypt this field
     */
    boolean encrypt() default true;
}

/**
 * Encryption type enum
 */
enum EncryptionType {
    /**
     * AES encryption
     */
    AES,
    
    /**
     * RSA encryption (for future use)
     */
    RSA
}