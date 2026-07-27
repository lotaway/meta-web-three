package com.metawebthree.wallet.infrastructure.solana;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.wallet.domain.entity.SolanaKeypair;
import com.metawebthree.wallet.infrastructure.persistence.repository.SolanaKeypairMapper;
import jakarta.annotation.PostConstruct;
import org.bouncycastle.crypto.params.Ed25519PrivateKeyParameters;
import org.bouncycastle.crypto.params.Ed25519PublicKeyParameters;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import javax.crypto.Cipher;
import javax.crypto.spec.GCMParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import java.security.SecureRandom;
import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

@Component
public class SolanaWalletManager {

    private static final Logger log = LoggerFactory.getLogger(SolanaWalletManager.class);
    private static final String AES_ALGORITHM = "AES/GCM/NoPadding";
    private static final int GCM_TAG_LENGTH = 128;
    private static final int IV_LENGTH = 12;

    private final SolanaKeypairMapper keypairMapper;
    private final byte[] masterKey;

    public SolanaWalletManager(SolanaKeypairMapper keypairMapper) {
        this.keypairMapper = keypairMapper;
        String envKey = System.getenv("SOLANA_ENCRYPTION_KEY");
        if (envKey != null && envKey.length() == 64) {
            this.masterKey = hexToBytes(envKey);
        } else {
            byte[] generated = new byte[32];
            new SecureRandom().nextBytes(generated);
            this.masterKey = generated;
            log.warn("SOLANA_ENCRYPTION_KEY not set or invalid; using randomly generated key. " +
                "Set a 64-char hex key via env var SOLANA_ENCRYPTION_KEY for production.");
        }
    }

    @PostConstruct
    public void init() {
        log.info("SolanaWalletManager initialized with AES-256-GCM encryption");
    }

    public String generateWallet() {
        byte[] seed = new byte[32];
        new SecureRandom().nextBytes(seed);
        Ed25519PrivateKeyParameters privKey = new Ed25519PrivateKeyParameters(seed, 0);
        Ed25519PublicKeyParameters pubKey = privKey.generatePublicKey();
        String address = SolanaContractClient.base58Encode(pubKey.getEncoded());

        storeKeypair(address, seed);
        log.info("Generated new Solana wallet: {}", address);
        return address;
    }

    public String importWallet(String privateKeyB58) {
        byte[] keypair = SolanaContractClient.decodeBase58(privateKeyB58);
        byte[] seed = new byte[32];
        System.arraycopy(keypair, 0, seed, 0, 32);
        byte[] publicKey = new byte[32];
        System.arraycopy(keypair, 32, publicKey, 0, 32);
        String address = SolanaContractClient.base58Encode(publicKey);

        storeKeypair(address, seed);
        log.info("Imported Solana wallet: {}", address);
        return address;
    }

    public byte[] getPrivateKey(String address) {
        SolanaKeypair entity = keypairMapper.selectOne(
            new LambdaQueryWrapper<SolanaKeypair>()
                .eq(SolanaKeypair::getAddress, address));
        if (entity == null) {
            throw new IllegalArgumentException("Wallet not found: " + address);
        }
        return decryptKey(entity.getEncryptedPrivateKey(), entity.getIv());
    }

    public List<String> listWallets() {
        return keypairMapper.selectList(null).stream()
            .map(SolanaKeypair::getAddress)
            .collect(Collectors.toList());
    }

    public boolean hasWallet(String address) {
        return keypairMapper.selectCount(
            new LambdaQueryWrapper<SolanaKeypair>()
                .eq(SolanaKeypair::getAddress, address)) > 0;
    }

    private void storeKeypair(String address, byte[] seed) {
        SolanaKeypair existing = keypairMapper.selectOne(
            new LambdaQueryWrapper<SolanaKeypair>()
                .eq(SolanaKeypair::getAddress, address));
        if (existing != null) {
            return;
        }

        byte[] iv = new byte[IV_LENGTH];
        new SecureRandom().nextBytes(iv);
        String encrypted = encryptKey(seed, iv);

        SolanaKeypair entity = new SolanaKeypair();
        entity.setAddress(address);
        entity.setEncryptedPrivateKey(encrypted);
        entity.setIv(bytesToHex(iv));
        entity.setCreatedAt(LocalDateTime.now());
        entity.setUpdatedAt(LocalDateTime.now());
        keypairMapper.insert(entity);
    }

    private String encryptKey(byte[] seed, byte[] iv) {
        try {
            Cipher cipher = Cipher.getInstance(AES_ALGORITHM);
            SecretKeySpec keySpec = new SecretKeySpec(masterKey, "AES");
            GCMParameterSpec gcmSpec = new GCMParameterSpec(GCM_TAG_LENGTH, iv);
            cipher.init(Cipher.ENCRYPT_MODE, keySpec, gcmSpec);
            byte[] ciphertext = cipher.doFinal(seed);
            return bytesToHex(ciphertext);
        } catch (Exception e) {
            throw new RuntimeException("Failed to encrypt private key", e);
        }
    }

    private byte[] decryptKey(String encryptedHex, String ivHex) {
        try {
            byte[] iv = hexToBytes(ivHex);
            byte[] ciphertext = hexToBytes(encryptedHex);
            Cipher cipher = Cipher.getInstance(AES_ALGORITHM);
            SecretKeySpec keySpec = new SecretKeySpec(masterKey, "AES");
            GCMParameterSpec gcmSpec = new GCMParameterSpec(GCM_TAG_LENGTH, iv);
            cipher.init(Cipher.DECRYPT_MODE, keySpec, gcmSpec);
            return cipher.doFinal(ciphertext);
        } catch (Exception e) {
            throw new RuntimeException("Failed to decrypt private key", e);
        }
    }

    static String bytesToHex(byte[] bytes) {
        StringBuilder sb = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) sb.append(String.format("%02x", b & 0xff));
        return sb.toString();
    }

    static byte[] hexToBytes(String hex) {
        int len = hex.length();
        byte[] bytes = new byte[len / 2];
        for (int i = 0; i < len; i += 2)
            bytes[i / 2] = (byte) ((Character.digit(hex.charAt(i), 16) << 4)
                + Character.digit(hex.charAt(i + 1), 16));
        return bytes;
    }
}
