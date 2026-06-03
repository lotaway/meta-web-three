package com.metawebthree.common.security;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.springframework.stereotype.Component;

import java.lang.reflect.Field;
import java.util.List;

/**
 * Aspect for automatically encrypting/decrypting fields annotated with @EncryptField
 * Applies to entity save/update operations and query operations
 */
@Slf4j
@Aspect
@Component
@RequiredArgsConstructor
public class EncryptionAspect {
    
    private final EncryptionService encryptionService;
    
    /**
     * Encrypt fields before saving entity
     * Applies to methods that save or update entities
     */
    @Around("@annotation(org.springframework.transaction.annotation.Transactional) && " +
            "(execution(* com.metawebthree..*.save*(..)) || " +
            "execution(* com.metawebthree..*.update*(..)) || " +
            "execution(* com.metawebthree..*.create*(..)))")
    public Object encryptBeforeSave(ProceedingJoinPoint joinPoint) throws Throwable {
        Object[] args = joinPoint.getArgs();
        
        // Encrypt sensitive fields in arguments
        for (Object arg : args) {
            if (arg != null) {
                encryptFields(arg);
            }
        }
        
        // Proceed with method execution
        Object result = joinPoint.proceed();
        
        // Decrypt sensitive fields in result (for returning to caller)
        if (result != null) {
            decryptFields(result);
        }
        
        return result;
    }
    
    /**
     * Decrypt fields after querying entities
     * Applies to methods that query entities
     */
    @Around("execution(* com.metawebthree..*.find*(..)) || " +
            "execution(* com.metawebthree..*.get*(..)) || " +
            "execution(* com.metawebthree..*.query*(..)) || " +
            "execution(* com.metawebthree..*.list*(..)) || " +
            "execution(* com.metawebthree..*.search*(..))")
    public Object decryptAfterQuery(ProceedingJoinPoint joinPoint) throws Throwable {
        // Proceed with method execution
        Object result = joinPoint.proceed();
        
        // Decrypt sensitive fields in result
        if (result != null) {
            if (result instanceof List) {
                List<?> list = (List<?>) result;
                for (Object item : list) {
                    if (item != null) {
                        decryptFields(item);
                    }
                }
            } else {
                decryptFields(result);
            }
        }
        
        return result;
    }
    
    /**
     * Encrypt fields annotated with @EncryptField
     */
    private void encryptFields(Object obj) {
        try {
            Field[] fields = obj.getClass().getDeclaredFields();
            for (Field field : fields) {
                if (field.isAnnotationPresent(EncryptField.class)) {
                    EncryptField annotation = field.getAnnotation(EncryptField.class);
                    if (annotation.encrypt()) {
                        field.setAccessible(true);
                        Object value = field.get(obj);
                        if (value != null && value instanceof String) {
                            String plaintext = (String) value;
                            String encrypted = encryptionService.encrypt(plaintext);
                            field.set(obj, encrypted);
                            log.debug("Encrypted field {} for class {}", field.getName(), obj.getClass().getSimpleName());
                        }
                    }
                }
            }
        } catch (Exception e) {
            log.error("Failed to encrypt fields for object: {}", obj, e);
        }
    }
    
    /**
     * Decrypt fields annotated with @EncryptField
     */
    private void decryptFields(Object obj) {
        try {
            Field[] fields = obj.getClass().getDeclaredFields();
            for (Field field : fields) {
                if (field.isAnnotationPresent(EncryptField.class)) {
                    EncryptField annotation = field.getAnnotation(EncryptField.class);
                    if (annotation.encrypt()) {
                        field.setAccessible(true);
                        Object value = field.get(obj);
                        if (value != null && value instanceof String) {
                            String ciphertext = (String) value;
                            String decrypted = encryptionService.decrypt(ciphertext);
                            field.set(obj, decrypted);
                            log.debug("Decrypted field {} for class {}", field.getName(), obj.getClass().getSimpleName());
                        }
                    }
                }
            }
        } catch (Exception e) {
            log.error("Failed to decrypt fields for object: {}", obj, e);
        }
    }
}