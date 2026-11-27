package com.metawebthree.wallet.application.service;

import com.metawebthree.wallet.infrastructure.solana.SolanaContractClient;
import com.metawebthree.wallet.infrastructure.solana.SolanaWalletManager;
import org.springframework.stereotype.Service;

/**
 * Program treasury operations: initialize the token-manager / treasury account,
 * deposit into it and withdraw from it. All three map 1:1 to contract instructions.
 */
@Service
public class SolanaTreasuryService {

    private final SolanaContractClient contractClient;
    private final SolanaWalletManager walletManager;

    public SolanaTreasuryService(SolanaContractClient contractClient, SolanaWalletManager walletManager) {
        this.contractClient = contractClient;
        this.walletManager = walletManager;
    }

    public String initialize(String signerAddress, String mintAddress) {
        requireAddress(signerAddress, "signerAddress");
        requireAddress(mintAddress, "mintAddress");

        byte[] signer = SolanaContractClient.decodeBase58(signerAddress);
        byte[] mint = SolanaContractClient.decodeBase58(mintAddress);
        byte[] privateKey = walletManager.getPrivateKey(signerAddress);
        return contractClient.initialize(signer, mint, privateKey);
    }

    public String deposit(String signerAddress, String mintAddress, Long amount) {
        long value = requireAmount(amount);
        byte[] signer = SolanaContractClient.decodeBase58(signerAddress);
        byte[] mint = SolanaContractClient.decodeBase58(mintAddress);
        byte[] privateKey = walletManager.getPrivateKey(signerAddress);
        return contractClient.deposit(signer, mint, value, privateKey);
    }

    public String withdraw(String signerAddress, String mintAddress, Long amount) {
        long value = requireAmount(amount);
        byte[] signer = SolanaContractClient.decodeBase58(signerAddress);
        byte[] mint = SolanaContractClient.decodeBase58(mintAddress);
        byte[] privateKey = walletManager.getPrivateKey(signerAddress);
        return contractClient.withdraw(signer, mint, value, privateKey);
    }

    private static long requireAmount(Long amount) {
        if (amount == null || amount <= 0) {
            throw new IllegalArgumentException("Amount must be positive");
        }
        return amount;
    }

    private static void requireAddress(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " is required");
        }
    }
}
