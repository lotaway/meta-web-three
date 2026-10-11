package com.metawebthree.wallet.application.service;

import com.metawebthree.wallet.application.dto.CouponDTO;
import com.metawebthree.wallet.infrastructure.solana.SolanaContractClient;
import com.metawebthree.wallet.infrastructure.solana.SolanaWalletManager;
import org.springframework.stereotype.Service;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Service
public class SolanaCouponService {

    private final SolanaContractClient contractClient;
    private final SolanaWalletManager walletManager;

    public SolanaCouponService(SolanaContractClient contractClient, SolanaWalletManager walletManager) {
        this.contractClient = contractClient;
        this.walletManager = walletManager;
    }

    public CouponDTO createCoupon(String authorityAddress, String mintAddress, Long discountAmount,
                                   Long maxUses, Long expiry) {
        if (discountAmount == null || discountAmount <= 0) {
            throw new IllegalArgumentException("Discount amount must be positive");
        }
        if (maxUses == null || maxUses <= 0) {
            throw new IllegalArgumentException("Max uses must be positive");
        }
        if (expiry == null || expiry <= 0) {
            throw new IllegalArgumentException("Expiry must be a future timestamp");
        }

        byte[] authority = SolanaContractClient.decodeBase58(authorityAddress);
        byte[] mint = SolanaContractClient.decodeBase58(mintAddress);
        byte[] privateKey = walletManager.getPrivateKey(authorityAddress);

        byte[] merkleRoot = computeMerkleRoot(authority);
        String txSig = contractClient.createCoupon(authority, mint, discountAmount, maxUses, merkleRoot, expiry, privateKey);

        byte[] coupon = contractClient.deriveCouponAddress(authority);
        return new CouponDTO(
            SolanaContractClient.base58Encode(coupon),
            authorityAddress,
            mintAddress,
            discountAmount,
            maxUses,
            0L,
            expiry,
            txSig
        );
    }

    public CouponDTO redeemCoupon(String authorityAddress, String userAddress, String mintAddress, List<String> proofHex) {
        byte[] authority = SolanaContractClient.decodeBase58(authorityAddress);
        byte[] user = SolanaContractClient.decodeBase58(userAddress);
        byte[] mint = SolanaContractClient.decodeBase58(mintAddress);
        byte[] privateKey = walletManager.getPrivateKey(userAddress);

        List<byte[]> proof = new ArrayList<>();
        if (proofHex != null) {
            for (String hex : proofHex) {
                proof.add(hexToBytes(hex));
            }
        }

        String txSig = contractClient.redeemCoupon(authority, user, mint, privateKey, proof);

        byte[] coupon = contractClient.deriveCouponAddress(authority);
        return new CouponDTO(
            SolanaContractClient.base58Encode(coupon),
            authorityAddress,
            mintAddress,
            0L, 0L, 0L, 0L, txSig
        );
    }

    private byte[] computeMerkleRoot(byte[] user) {
        try {
            MessageDigest sha = MessageDigest.getInstance("SHA-256");
            return sha.digest(user);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
    }

    private byte[] hexToBytes(String hex) {
        int len = hex.length();
        byte[] bytes = new byte[len / 2];
        for (int i = 0; i < len; i += 2)
            bytes[i / 2] = (byte) ((Character.digit(hex.charAt(i), 16) << 4)
                + Character.digit(hex.charAt(i + 1), 16));
        return bytes;
    }
}
