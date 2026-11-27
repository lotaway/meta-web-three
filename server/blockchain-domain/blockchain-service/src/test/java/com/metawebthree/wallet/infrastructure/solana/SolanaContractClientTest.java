package com.metawebthree.wallet.infrastructure.solana;

import org.bouncycastle.math.ec.rfc8032.Ed25519;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SolanaContractClientTest {

    private static final String PROGRAM_ID = "FiSvBZn8sSV3r8xC6PcEC59D8oBeDUj2sdS9Jdppq9oW";
    private static final byte[] AUTHORITY = bytesZeroTo31();

    private SolanaContractClient client;

    @BeforeEach
    void setUp() {
        // PDA derivation is pure math; the RPC client is never touched.
        client = new SolanaContractClient(null, properties(PROGRAM_ID));
    }

    private static SolanaProperties properties(String programId) {
        // Test fixtures only: production defaults live in application.yml.
        SolanaProperties properties = new SolanaProperties();
        properties.setProgramId(programId);
        properties.setPaymentMint("So11111111111111111111111111111111111111112");
        SolanaProperties.Programs programs = new SolanaProperties.Programs();
        programs.setToken("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA");
        programs.setAssociatedToken("ATokenGPvbdGVxr1b2hvZbsiqW5xr25ix9f2JtAbjv9w");
        programs.setMetadata("metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s");
        programs.setSystem("11111111111111111111111111111111");
        programs.setRent("SysvarRent111111111111111111111111111111111");
        properties.setPrograms(programs);
        return properties;
    }

    @Test
    void activityPda_matchesSolanaReference() {
        SolanaContractClient.Pda pda = client.findProgramAddress(
            List.of("activity".getBytes(StandardCharsets.UTF_8), AUTHORITY), null);

        assertEquals("CmgA2W4PWu93H372KZdQJaDaHHnbn8tjyxK2gYEskAr2",
            SolanaContractClient.base58Encode(pda.address()));
        assertEquals(254, pda.bump());
        assertArrayEquals(pda.address(), client.deriveActivityAddress(AUTHORITY));
    }

    @Test
    void activityPoolPda_usesPoolSeed_notAta() {
        SolanaContractClient.Pda pda = client.findProgramAddress(
            List.of("activity".getBytes(StandardCharsets.UTF_8), AUTHORITY,
                "pool".getBytes(StandardCharsets.UTF_8)), null);

        assertEquals("FNWCq1jpzVZEqYnSbHLPqhcKt3cnDXSU3qiZWhF3742B",
            SolanaContractClient.base58Encode(pda.address()));
        assertEquals(255, pda.bump());
        assertArrayEquals(pda.address(), client.deriveActivityPool(AUTHORITY));

        // The pool PDA must NOT be the (wrong) ATA-of-activity the old code used.
        assertFalse(java.util.Arrays.equals(pda.address(),
            client.findAssociatedTokenAddress(AUTHORITY, client.deriveActivityAddress(AUTHORITY))));
    }

    @Test
    void listingPda_matchesSolanaReference() {
        SolanaContractClient.Pda pda = client.findProgramAddress(
            List.of("listing".getBytes(StandardCharsets.UTF_8), AUTHORITY, AUTHORITY), null);

        assertEquals("3SHuEctD5X6qApU2weiQH9aE5XPekqrJ53sdHykDUkbe",
            SolanaContractClient.base58Encode(pda.address()));
        assertEquals(254, pda.bump());
        assertArrayEquals(pda.address(), client.deriveListingAddress(AUTHORITY, AUTHORITY));
    }

    @Test
    void findProgramAddress_includesBumpInHash() {
        // Regression for the original bug: the bump byte was omitted, so every
        // iteration hashed identically (and the off-curve check used the wrong
        // sign-bit heuristic). The result must equal sha256(seeds + bump +
        // programId + "ProgramDerivedAddress") and differ from the bump-less hash.
        byte[] programId = SolanaContractClient.decodeBase58(PROGRAM_ID);
        List<byte[]> seeds = List.of("activity".getBytes(StandardCharsets.UTF_8), AUTHORITY);
        SolanaContractClient.Pda pda = client.findProgramAddress(seeds, programId);

        assertArrayEquals(sha256(concat(seeds, new byte[]{(byte) pda.bump()}, programId,
            "ProgramDerivedAddress".getBytes(StandardCharsets.UTF_8))), pda.address());

        byte[] withoutBump = sha256(concat(seeds, programId,
            "ProgramDerivedAddress".getBytes(StandardCharsets.UTF_8)));
        assertFalse(java.util.Arrays.equals(withoutBump, pda.address()));
    }

    @Test
    void findProgramAddress_returnsOffCurvePoint() {
        // A valid PDA is off the curve. validatePublicKeyPartial (not Full)
        // mirrors Solana's CompressedEdwardsY::decompress() check.
        byte[] activity = client.deriveActivityAddress(AUTHORITY);
        byte[] pool = client.deriveActivityPool(AUTHORITY);
        byte[] listing = client.deriveListingAddress(AUTHORITY, AUTHORITY);

        assertFalse(Ed25519.validatePublicKeyPartial(activity, 0));
        assertFalse(Ed25519.validatePublicKeyPartial(pool, 0));
        assertFalse(Ed25519.validatePublicKeyPartial(listing, 0));
    }

    @Test
    void associatedTokenAddress_matchesSolanaReference() {
        // SPL ATA = PDA([owner, token_program, mint], ATA program). The old code
        // hashed seeds without bump/programId/marker and produced a wrong address.
        byte[] mint = new byte[32];
        for (int i = 0; i < 32; i++) mint[i] = (byte) (i + 32);
        byte[] ata = client.findAssociatedTokenAddress(mint, AUTHORITY);

        assertEquals("D1FwhcxiNcQKMvbVANWRZesRyM5cpdXiLpK9qMeuZe2k",
            SolanaContractClient.base58Encode(ata));
        assertFalse(Ed25519.validatePublicKeyPartial(ata, 0));
    }

    @Test
    void nftMintPda_usesMintSeed_notSftSeed() {
        // create_token_and_nft uses [b"mint", authority]; SFT uses
        // [b"sft", name, authority]. They must be different addresses.
        byte[] nftMint = client.deriveNftMint(AUTHORITY);
        assertEquals("7Wb1gGFKvbburjjvQKH9DnuAVwTCGMJMuXVZ3nWCPijy",
            SolanaContractClient.base58Encode(nftMint));
        assertFalse(java.util.Arrays.equals(nftMint, client.deriveSftMint("art", AUTHORITY)));
        assertFalse(Ed25519.validatePublicKeyPartial(nftMint, 0));
    }

    @Test
    void treasuryPdas_matchSolanaReference() {
        // initialize / deposit / withdraw use these two PDAs.
        assertEquals("ADqxX3okMerKJyaQLcYL1BHearPkXzobJehcwe9Xcwb6",
            SolanaContractClient.base58Encode(client.deriveTokenManager()));

        byte[] mint = new byte[32];
        for (int i = 0; i < 32; i++) mint[i] = (byte) (i + 32);
        assertEquals("7FzEhH1T6LVPFSh3f6BFVxxNygxj4iTxu6H4TjcRKPKG",
            SolanaContractClient.base58Encode(client.deriveProgramTokenAccount(mint)));

        assertFalse(Ed25519.validatePublicKeyPartial(client.deriveTokenManager(), 0));
        assertFalse(Ed25519.validatePublicKeyPartial(client.deriveProgramTokenAccount(mint), 0));
    }

    @Test
    void programId_isInjectedAndAffectsPda() {
        SolanaContractClient systemProgramClient =
            new SolanaContractClient(null, properties("11111111111111111111111111111111"));

        assertFalse(java.util.Arrays.equals(
            client.deriveActivityAddress(AUTHORITY),
            systemProgramClient.deriveActivityAddress(AUTHORITY)));
        assertTrue(SolanaContractClient.base58Encode(client.deriveActivityAddress(AUTHORITY)).length() >= 32);
    }

    private static byte[] bytesZeroTo31() {
        byte[] out = new byte[32];
        for (int i = 0; i < 32; i++) out[i] = (byte) i;
        return out;
    }

    private static byte[] concat(byte[]... parts) {
        int len = 0;
        for (byte[] p : parts) len += p.length;
        ByteBuffer buf = ByteBuffer.allocate(len);
        for (byte[] p : parts) buf.put(p);
        return buf.array();
    }

    private static byte[] concat(List<byte[]> seeds, byte[]... tail) {
        int len = 0;
        for (byte[] s : seeds) len += s.length;
        for (byte[] t : tail) len += t.length;
        ByteBuffer buf = ByteBuffer.allocate(len);
        for (byte[] s : seeds) buf.put(s);
        for (byte[] t : tail) buf.put(t);
        return buf.array();
    }

    private static byte[] sha256(byte[] input) {
        try {
            return MessageDigest.getInstance("SHA-256").digest(input);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
