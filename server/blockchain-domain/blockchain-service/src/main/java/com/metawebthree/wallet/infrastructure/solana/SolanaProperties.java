package com.metawebthree.wallet.infrastructure.solana;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * Deployment-specific Solana values bound from {@code blockchain.solana.*} in
 * application.yml. There are deliberately NO default values here — all defaults
 * live in yml (with env-var overrides) so there is a single source of truth.
 */
@Component
@ConfigurationProperties(prefix = "blockchain.solana")
public class SolanaProperties {

    /** declare_id! of the deployed Anchor program (contracts/solana/lib.rs). */
    private String programId;

    /** Payment mint for activities / marketplace when the caller does not supply one. */
    private String paymentMint;

    /** AES-256 master key (64 hex chars) for managed wallet private keys; empty = dev only. */
    private String encryptionKey;

    /** Default activity reward split in basis points [1st, 2nd, 3rd]. */
    private int[] defaultRewardPercentages;

    private Programs programs = new Programs();

    public static class Programs {
        private String token;
        private String associatedToken;
        private String metadata;
        private String system;
        private String rent;

        public String getToken() { return token; }
        public void setToken(String token) { this.token = token; }
        public String getAssociatedToken() { return associatedToken; }
        public void setAssociatedToken(String associatedToken) { this.associatedToken = associatedToken; }
        public String getMetadata() { return metadata; }
        public void setMetadata(String metadata) { this.metadata = metadata; }
        public String getSystem() { return system; }
        public void setSystem(String system) { this.system = system; }
        public String getRent() { return rent; }
        public void setRent(String rent) { this.rent = rent; }
    }

    /** Fail fast with a useful message when deployment config is missing. */
    public void requireDeployedAddresses() {
        require("program-id", programId);
        require("payment-mint", paymentMint);
        if (programs == null) {
            throw new IllegalArgumentException("Missing config: blockchain.solana.programs");
        }
        require("programs.token", programs.token);
        require("programs.associated-token", programs.associatedToken);
        require("programs.metadata", programs.metadata);
        require("programs.system", programs.system);
        require("programs.rent", programs.rent);
    }

    private static void require(String key, String value) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("Missing config: blockchain.solana." + key);
        }
    }

    public String getProgramId() { return programId; }
    public void setProgramId(String programId) { this.programId = programId; }
    public String getPaymentMint() { return paymentMint; }
    public void setPaymentMint(String paymentMint) { this.paymentMint = paymentMint; }
    public String getEncryptionKey() { return encryptionKey; }
    public void setEncryptionKey(String encryptionKey) { this.encryptionKey = encryptionKey; }
    public int[] getDefaultRewardPercentages() { return defaultRewardPercentages; }
    public void setDefaultRewardPercentages(int[] defaultRewardPercentages) {
        this.defaultRewardPercentages = defaultRewardPercentages;
    }
    public Programs getPrograms() { return programs; }
    public void setPrograms(Programs programs) { this.programs = programs; }
}
