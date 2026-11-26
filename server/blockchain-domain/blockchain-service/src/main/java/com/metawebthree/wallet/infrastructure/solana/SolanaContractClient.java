package com.metawebthree.wallet.infrastructure.solana;

import org.bouncycastle.crypto.signers.Ed25519Signer;
import org.bouncycastle.crypto.params.Ed25519PrivateKeyParameters;
import org.bouncycastle.math.ec.rfc8032.Ed25519;
import org.springframework.stereotype.Component;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.*;

@Component
public class SolanaContractClient {

    private static final String ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    private static final int[] BASE58_DECODE = new int[128];
    static {
        Arrays.fill(BASE58_DECODE, -1);
        for (int i = 0; i < ALPHABET.length(); i++) BASE58_DECODE[ALPHABET.charAt(i)] = i;
    }

    // All external program ids come from blockchain.solana.* configuration
    // (SolanaProperties), never hardcoded here.
    private final byte[] programId;
    private final byte[] tokenProgramId;
    private final byte[] associatedTokenProgramId;
    private final byte[] tokenMetadataProgramId;
    private final byte[] systemProgramId;
    private final byte[] rentSysvarId;

    private final SolanaRpcClient rpcClient;

    public SolanaContractClient(SolanaRpcClient rpcClient, SolanaProperties properties) {
        this.rpcClient = rpcClient;
        properties.requireDeployedAddresses();
        this.programId = decodeBase58(properties.getProgramId());
        SolanaProperties.Programs programs = properties.getPrograms();
        this.tokenProgramId = decodeBase58(programs.getToken());
        this.associatedTokenProgramId = decodeBase58(programs.getAssociatedToken());
        this.tokenMetadataProgramId = decodeBase58(programs.getMetadata());
        this.systemProgramId = decodeBase58(programs.getSystem());
        this.rentSysvarId = decodeBase58(programs.getRent());
    }

    // ──────────────────────────────────────────────
    // PDA Derivation
    // ──────────────────────────────────────────────

    public record Pda(byte[] address, int bump) {}

    public Pda findProgramAddress(List<byte[]> seeds, byte[] programIdOverride) {
        byte[] programIdBytes = programIdOverride != null ? programIdOverride : this.programId;
        for (int bump = 255; bump >= 0; bump--) {
            try {
                MessageDigest sha = MessageDigest.getInstance("SHA-256");
                for (byte[] seed : seeds) sha.update(seed);
                // Anchor's find_program_address appends the bump byte BEFORE the
                // program id and the "ProgramDerivedAddress" marker. Omitting the
                // bump makes every iteration hash identically and yields a wrong PDA.
                sha.update((byte) bump);
                sha.update(programIdBytes);
                sha.update("ProgramDerivedAddress".getBytes(StandardCharsets.UTF_8));
                byte[] hash = sha.digest();
                // A valid PDA MUST be off the ed25519 curve. Solana checks this
                // with CompressedEdwardsY::decompress() (decompressable => on
                // curve => skip). BC's validatePublicKeyPartial mirrors that
                // check; validatePublicKeyFull additionally enforces subgroup
                // order, which Solana's PDA check does NOT do.
                if (!Ed25519.validatePublicKeyPartial(hash, 0)) {
                    return new Pda(hash, bump);
                }
            } catch (NoSuchAlgorithmException e) {
                throw new RuntimeException("SHA-256 not available", e);
            }
        }
        throw new RuntimeException("Unable to find valid PDA");
    }

    public byte[] deriveMintAddress(String seedPrefix, String name, byte[] authority) {
        return findProgramAddress(List.of(seedPrefix.getBytes(StandardCharsets.UTF_8),
            name.getBytes(StandardCharsets.UTF_8), authority), programId).address();
    }

    public byte[] deriveTokenMint(String name, byte[] authority) {
        return deriveMintAddress("token", name, authority);
    }

    public byte[] deriveSftMint(String name, byte[] authority) {
        return deriveMintAddress("sft", name, authority);
    }

    /** NFT mint PDA seeds [b"mint", authority] (CreateTokenAndNFT in lib.rs:108). */
    public byte[] deriveNftMint(byte[] authority) {
        return findProgramAddress(List.of("mint".getBytes(StandardCharsets.UTF_8), authority), programId).address();
    }

    public byte[] deriveListingAddress(byte[] seller, byte[] mint) {
        return findProgramAddress(List.of("listing".getBytes(StandardCharsets.UTF_8), seller, mint), programId).address();
    }

    public byte[] deriveListingEscrow(byte[] mint) {
        return findProgramAddress(List.of("listing_escrow".getBytes(StandardCharsets.UTF_8), mint), programId).address();
    }

    public byte[] deriveActivityAddress(byte[] authority) {
        return findProgramAddress(List.of("activity".getBytes(StandardCharsets.UTF_8), authority), programId).address();
    }

    // Mirrors the on-chain PDA seeds [ACTIVITY, authority, b"pool"] used by
    // participate_activity / claim_reward (lib.rs:452,487). It is NOT an ATA.
    public byte[] deriveActivityPool(byte[] authority) {
        return findProgramAddress(
            List.of("activity".getBytes(StandardCharsets.UTF_8), authority, "pool".getBytes(StandardCharsets.UTF_8)), programId).address();
    }

    public byte[] deriveCommissionGraph(byte[] target) {
        return findProgramAddress(List.of("commission".getBytes(StandardCharsets.UTF_8), target), programId).address();
    }

    public byte[] deriveCouponAddress(byte[] authority) {
        return findProgramAddress(List.of("coupon".getBytes(StandardCharsets.UTF_8), authority), programId).address();
    }

    public byte[] deriveCouponPoolAddress(byte[] authority) {
        return findProgramAddress(List.of("coupon".getBytes(StandardCharsets.UTF_8), authority, "pool".getBytes(StandardCharsets.UTF_8)), programId).address();
    }

    public byte[] deriveMetadataAddress(byte[] mint) {
        return findProgramAddress(List.of("metadata".getBytes(StandardCharsets.UTF_8), tokenMetadataProgramId, mint), tokenMetadataProgramId).address();
    }

    // ──────────────────────────────────────────────
    // Anchor instruction building
    // ──────────────────────────────────────────────

    private byte[] discriminator(String methodName) {
        try {
            MessageDigest sha = MessageDigest.getInstance("SHA-256");
            sha.update(("global:" + methodName).getBytes(StandardCharsets.UTF_8));
            byte[] hash = sha.digest();
            return Arrays.copyOf(hash, 8);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
    }

    private byte[] borshString(String s) {
        byte[] utf8 = s.getBytes(StandardCharsets.UTF_8);
        ByteBuffer buf = ByteBuffer.allocate(4 + utf8.length).order(ByteOrder.LITTLE_ENDIAN);
        buf.putInt(utf8.length);
        buf.put(utf8);
        return buf.array();
    }

    private byte[] borshU64(long v) {
        return ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(v).array();
    }

    private byte[] borshI64(long v) {
        return ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(v).array();
    }

    private byte[] borshU16Array(int[] values) {
        ByteBuffer buf = ByteBuffer.allocate(values.length * 2).order(ByteOrder.LITTLE_ENDIAN);
        for (int v : values) buf.putShort((short) v);
        return buf.array();
    }

    private byte[] borshU8Array(byte[] values) {
        ByteBuffer buf = ByteBuffer.allocate(4 + values.length).order(ByteOrder.LITTLE_ENDIAN);
        buf.putInt(values.length);
        buf.put(values);
        return buf.array();
    }

    private byte[] borshU8Array32(byte[] values) {
        if (values.length != 32) throw new IllegalArgumentException("Expected 32 bytes");
        return values;
    }

    private byte[] borshU8(int v) {
        return new byte[]{(byte) v};
    }

    public record AccountMeta(byte[] pubkey, boolean isSigner, boolean isWritable) {}
    public record SolInstruction(byte[] programId, List<AccountMeta> accounts, byte[] data) {}

    public String buildAndSend(List<SolInstruction> instructions, List<byte[]> privateKeys) {
        try {
            // Merge duplicate accounts and collect combined flags
            Map<String, Boolean[]> merged = new LinkedHashMap<>();
            for (SolInstruction ix : instructions) {
                merged.put(base58Encode(ix.programId()), new Boolean[]{false, false});
                for (AccountMeta meta : ix.accounts()) {
                    merged.merge(base58Encode(meta.pubkey()), new Boolean[]{meta.isSigner(), meta.isWritable()},
                        (a, b) -> new Boolean[]{a[0] || b[0], a[1] && b[1]});
                }
            }

            // Sort: writable signers → readonly signers → writable non-signers → readonly non-signers
            List<Map.Entry<String, Boolean[]>> sorted = new ArrayList<>(merged.entrySet());
            sorted.sort((a, b) -> {
                boolean aSigner = a.getValue()[0], aWritable = a.getValue()[1];
                boolean bSigner = b.getValue()[0], bWritable = b.getValue()[1];
                int catA = aSigner ? (aWritable ? 0 : 1) : (aWritable ? 2 : 3);
                int catB = bSigner ? (bWritable ? 0 : 1) : (bWritable ? 2 : 3);
                return Integer.compare(catA, catB);
            });

            // Build sorted account key list and index mapping
            List<byte[]> allPubkeys = new ArrayList<>();
            Map<String, Integer> keyIndex = new HashMap<>();
            int numRequiredSigs = 0, numReadonlySigned = 0, numReadonlyUnsigned = 0;

            for (var entry : sorted) {
                String keyStr = entry.getKey();
                boolean isSigner = entry.getValue()[0];
                boolean isWritable = entry.getValue()[1];
                keyIndex.put(keyStr, allPubkeys.size());
                allPubkeys.add(decodeBase58(keyStr));
                if (isSigner) {
                    numRequiredSigs++;
                    if (!isWritable) numReadonlySigned++;
                } else {
                    if (!isWritable) numReadonlyUnsigned++;
                }
            }

            Map<String, Object> bh = rpcClient.getLatestBlockhash();
            byte[] blockhash = decodeBase58((String) bh.get("blockhash"));

            ByteArrayOutputStream msg = new ByteArrayOutputStream();
            msg.write(numRequiredSigs);
            msg.write(numReadonlySigned);
            msg.write(numReadonlyUnsigned);
            writeCompactArrayU8x32(msg, allPubkeys);
            writeBytes(msg, blockhash);

            // Build compiled instructions
            byte[][] instructionsData = new byte[instructions.size()][];
            int[] programIds = new int[instructions.size()];
            int[][] accountIndexes = new int[instructions.size()][];

            for (int i = 0; i < instructions.size(); i++) {
                SolInstruction ix = instructions.get(i);
                programIds[i] = keyIndex.get(base58Encode(ix.programId()));
                instructionsData[i] = ix.data();
                accountIndexes[i] = ix.accounts().stream()
                    .mapToInt(a -> keyIndex.get(base58Encode(a.pubkey())))
                    .toArray();
            }

            writeCompactInstructions(msg, instructionsData, programIds, accountIndexes);

            byte[] messageBytes = msg.toByteArray();
            MessageDigest sha = MessageDigest.getInstance("SHA-256");
            byte[] messageHash = sha.digest(messageBytes);

            // Build signature list matching sorted signer order
            List<byte[]> signatures = new ArrayList<>();
            for (var entry : sorted) {
                if (!entry.getValue()[0]) continue;
                byte[] pk = decodeBase58(entry.getKey());
                boolean found = false;
                for (byte[] priv : privateKeys) {
                    if (getPublicKey(priv).equals(entry.getKey())) {
                        signatures.add(sign(priv, messageHash));
                        found = true;
                        break;
                    }
                }
                if (!found) signatures.add(new byte[64]);
            }

            // Serialize transaction: compact array of signatures + message
            ByteArrayOutputStream tx = new ByteArrayOutputStream();
            writeCompactArrayRaw(tx, signatures);
            tx.write(messageBytes);

            String base64Tx = java.util.Base64.getEncoder().encodeToString(tx.toByteArray());
            return rpcClient.sendTransaction(base64Tx);
        } catch (Exception e) {
            throw new RuntimeException("Failed to build and send transaction", e);
        }
    }

    private byte[] sign(byte[] privateKey, byte[] message) {
        Ed25519PrivateKeyParameters params = new Ed25519PrivateKeyParameters(privateKey, 0);
        Ed25519Signer signer = new Ed25519Signer();
        signer.init(true, params);
        signer.update(message, 0, message.length);
        return signer.generateSignature();
    }

    private String getPublicKey(byte[] privateKey) {
        Ed25519PrivateKeyParameters params = new Ed25519PrivateKeyParameters(privateKey, 0);
        return base58Encode(params.generatePublicKey().getEncoded());
    }

    // ──────────────────────────────────────────────
    // Serialization helpers
    // ──────────────────────────────────────────────

    private void writeCompactArrayU8x32(ByteArrayOutputStream bos, List<byte[]> items) {
        writeCompactU16Length(bos, items.size());
        for (byte[] item : items) writeBytes(bos, item);
    }

    private void writeCompactArrayRaw(ByteArrayOutputStream bos, List<byte[]> items) {
        writeCompactU16Length(bos, items.size());
        for (byte[] item : items) writeBytes(bos, item);
    }

    private void writeCompactInstructions(ByteArrayOutputStream bos, byte[][] data, int[] programIds, int[][] accountIdx) {
        writeCompactU16Length(bos, data.length);
        for (int i = 0; i < data.length; i++) {
            bos.write(programIds[i]);
            writeCompactU16Length(bos, accountIdx[i].length);
            for (int idx : accountIdx[i]) bos.write(idx);
            writeCompactU16Length(bos, data[i].length);
            writeBytes(bos, data[i]);
        }
    }

    private void writeCompactU16Length(ByteArrayOutputStream bos, int value) {
        if (value < 128) {
            bos.write(value);
        } else if (value < 16384) {
            bos.write((value & 0x7F) | 0x80);
            bos.write(value >> 7);
        } else {
            bos.write((value & 0x7F) | 0x80);
            bos.write(((value >> 7) & 0x7F) | 0x80);
            bos.write(value >> 14);
        }
    }

    private void writeBytes(ByteArrayOutputStream bos, byte[] bytes) {
        bos.write(bytes, 0, bytes.length);
    }

    // ──────────────────────────────────────────────
    // Account meta helpers for each instruction
    // ──────────────────────────────────────────────

    public record TokenInstruction(
        String name, String symbol, String uri, long supply,
        byte[] authority, byte[] mint, byte[] tokenAccount, byte[] metadata
    ) {}

    public String createToken(TokenInstruction ti, byte[] privateKey) {
        byte[] mint = deriveTokenMint(ti.name(), ti.authority());
        byte[] meta = deriveMetadataAddress(mint);
        byte[] ata = findAssociatedTokenAddress(mint, ti.authority());

        List<AccountMeta> accounts = List.of(
            new AccountMeta(ti.authority(), true, true),
            new AccountMeta(mint, false, true),
            new AccountMeta(ata, false, true),
            new AccountMeta(meta, false, true),
            new AccountMeta(tokenMetadataProgramId, false, false),
            new AccountMeta(tokenProgramId, false, false),
            new AccountMeta(associatedTokenProgramId, false, false),
            new AccountMeta(systemProgramId, false, false),
            new AccountMeta(rentSysvarId, false, false)
        );

        byte[] data = concat(discriminator("create_token"),
            borshString(ti.name()),
            borshString(ti.symbol()),
            borshString(ti.uri()),
            borshU64(ti.supply()));

        return buildAndSend(List.of(new SolInstruction(programId, accounts, data)), List.of(privateKey));
    }

    public String createSft(String name, String symbol, String uri, long supply, byte[] authority, byte[] privateKey) {
        byte[] mint = deriveSftMint(name, authority);
        byte[] meta = deriveMetadataAddress(mint);
        byte[] ata = findAssociatedTokenAddress(mint, authority);

        List<AccountMeta> accounts = List.of(
            new AccountMeta(authority, true, true),
            new AccountMeta(mint, false, true),
            new AccountMeta(ata, false, true),
            new AccountMeta(meta, false, true),
            new AccountMeta(tokenMetadataProgramId, false, false),
            new AccountMeta(tokenProgramId, false, false),
            new AccountMeta(associatedTokenProgramId, false, false),
            new AccountMeta(systemProgramId, false, false),
            new AccountMeta(rentSysvarId, false, false)
        );

        byte[] data = concat(discriminator("create_sft"),
            borshString(name), borshString(symbol), borshString(uri), borshU64(supply));

        return buildAndSend(List.of(new SolInstruction(programId, accounts, data)), List.of(privateKey));
    }

    /**
     * Dedicated NFT instruction (supply fixed at 1 on-chain). The mint is the
     * PDA [b"mint", authority] — not the SFT seed. Mirrors CreateTokenAndNFT
     * accounts (lib.rs:101) and create_token_and_nft args (name, symbol, uri).
     */
    public String createTokenAndNft(String name, String symbol, String uri, byte[] authority, byte[] privateKey) {
        byte[] mint = deriveNftMint(authority);
        byte[] meta = deriveMetadataAddress(mint);
        byte[] ata = findAssociatedTokenAddress(mint, authority);

        List<AccountMeta> accounts = List.of(
            new AccountMeta(authority, true, true),
            new AccountMeta(mint, false, true),
            new AccountMeta(ata, false, true),
            new AccountMeta(meta, false, true),
            new AccountMeta(tokenMetadataProgramId, false, false),
            new AccountMeta(tokenProgramId, false, false),
            new AccountMeta(associatedTokenProgramId, false, false),
            new AccountMeta(systemProgramId, false, false),
            new AccountMeta(rentSysvarId, false, false)
        );

        byte[] data = concat(discriminator("create_token_and_nft"),
            borshString(name), borshString(symbol), borshString(uri));

        return buildAndSend(List.of(new SolInstruction(programId, accounts, data)), List.of(privateKey));
    }

    public String mintTo(byte[] mint, byte[] receiver, long amount, byte[] authority, byte[] privateKey) {
        byte[] ata = findAssociatedTokenAddress(mint, receiver);

        List<AccountMeta> accounts = List.of(
            new AccountMeta(authority, true, true),
            new AccountMeta(mint, false, true),
            new AccountMeta(ata, false, true),
            new AccountMeta(receiver, false, false),
            new AccountMeta(tokenProgramId, false, false),
            new AccountMeta(associatedTokenProgramId, false, false),
            new AccountMeta(systemProgramId, false, false)
        );

        byte[] data = concat(discriminator("mint_to"), borshU64(amount));
        return buildAndSend(List.of(new SolInstruction(programId, accounts, data)), List.of(privateKey));
    }

    public String burnTokens(byte[] mint, byte[] owner, long amount, byte[] privateKey) {
        byte[] ata = findAssociatedTokenAddress(mint, owner);

        List<AccountMeta> accounts = List.of(
            new AccountMeta(owner, true, true),
            new AccountMeta(mint, false, true),
            new AccountMeta(ata, false, true),
            new AccountMeta(tokenProgramId, false, false)
        );

        byte[] data = concat(discriminator("burn_tokens"), borshU64(amount));
        return buildAndSend(List.of(new SolInstruction(programId, accounts, data)), List.of(privateKey));
    }

    public String listGood(byte[] seller, byte[] mint, byte[] paymentMint, long price, long listedAmount, byte[] privateKey) {
        byte[] listing = deriveListingAddress(seller, mint);
        byte[] escrow = deriveListingEscrow(mint);
        byte[] sellerTokenAccount = findAssociatedTokenAddress(mint, seller);

        List<AccountMeta> accounts = List.of(
            new AccountMeta(seller, true, true),
            new AccountMeta(listing, false, true),
            new AccountMeta(mint, false, false),
            new AccountMeta(paymentMint, false, false),
            new AccountMeta(sellerTokenAccount, false, true),
            new AccountMeta(escrow, false, true),
            new AccountMeta(tokenProgramId, false, false),
            new AccountMeta(associatedTokenProgramId, false, false),
            new AccountMeta(systemProgramId, false, false),
            new AccountMeta(rentSysvarId, false, false)
        );

        byte[] data = concat(discriminator("list_good"), borshU64(price), borshU64(listedAmount));
        return buildAndSend(List.of(new SolInstruction(programId, accounts, data)), List.of(privateKey));
    }

    public String buyGood(byte[] buyer, byte[] seller, byte[] mint, byte[] paymentMint, byte[] listing, byte[] privateKey) {
        byte[] escrow = deriveListingEscrow(mint);
        byte[] buyerPaymentAta = findAssociatedTokenAddress(paymentMint, buyer);
        byte[] sellerPaymentAta = findAssociatedTokenAddress(paymentMint, seller);
        byte[] buyerReceiveAta = findAssociatedTokenAddress(mint, buyer);

        List<AccountMeta> accounts = List.of(
            new AccountMeta(buyer, true, true),
            new AccountMeta(listing, false, true),
            new AccountMeta(seller, false, true),
            new AccountMeta(mint, false, false),
            new AccountMeta(paymentMint, false, false),
            new AccountMeta(buyerPaymentAta, false, true),
            new AccountMeta(sellerPaymentAta, false, true),
            new AccountMeta(buyerReceiveAta, false, true),
            new AccountMeta(escrow, false, true),
            new AccountMeta(tokenProgramId, false, false),
            new AccountMeta(associatedTokenProgramId, false, false),
            new AccountMeta(systemProgramId, false, false),
            new AccountMeta(rentSysvarId, false, false)
        );

        byte[] data = discriminator("buy_good");
        return buildAndSend(List.of(new SolInstruction(programId, accounts, data)), List.of(privateKey));
    }

    public String delistGood(byte[] seller, byte[] mint, byte[] paymentMint, byte[] privateKey) {
        byte[] listing = deriveListingAddress(seller, mint);
        byte[] escrow = deriveListingEscrow(mint);
        byte[] sellerTokenAccount = findAssociatedTokenAddress(mint, seller);

        // Must match DelistGood's 10 accounts exactly (lib.rs:507).
        List<AccountMeta> accounts = List.of(
            new AccountMeta(seller, true, true),
            new AccountMeta(listing, false, true),
            new AccountMeta(mint, false, false),
            new AccountMeta(paymentMint, false, false),
            new AccountMeta(sellerTokenAccount, false, true),
            new AccountMeta(escrow, false, true),
            new AccountMeta(tokenProgramId, false, false),
            new AccountMeta(associatedTokenProgramId, false, false),
            new AccountMeta(systemProgramId, false, false),
            new AccountMeta(rentSysvarId, false, false)
        );

        byte[] data = discriminator("delist_good");
        return buildAndSend(List.of(new SolInstruction(programId, accounts, data)), List.of(privateKey));
    }

    public String createActivity(byte[] authority, long startTime, long endTime, long entryFee, int[] rewardPcts, byte[] privateKey) {
        byte[] activity = deriveActivityAddress(authority);

        List<AccountMeta> accounts = List.of(
            new AccountMeta(authority, true, true),
            new AccountMeta(activity, false, true),
            new AccountMeta(systemProgramId, false, false),
            new AccountMeta(rentSysvarId, false, false)
        );

        byte[] data = concat(discriminator("create_activity"),
            borshI64(startTime), borshI64(endTime), borshU64(entryFee), borshU16Array(rewardPcts));
        return buildAndSend(List.of(new SolInstruction(programId, accounts, data)), List.of(privateKey));
    }

    public String participateActivity(byte[] participant, byte[] authority, byte[] paymentMint, byte[] privateKey) {
        byte[] activity = deriveActivityAddress(authority);
        byte[] participantTokenAccount = findAssociatedTokenAddress(paymentMint, participant);
        byte[] poolTokenAccount = deriveActivityPool(authority);

        List<AccountMeta> accounts = List.of(
            new AccountMeta(participant, true, true),
            new AccountMeta(activity, false, true),
            // Order must match ParticipateActivity: pool before participant ATA.
            new AccountMeta(poolTokenAccount, false, true),
            new AccountMeta(participantTokenAccount, false, true),
            new AccountMeta(paymentMint, false, false),
            new AccountMeta(tokenProgramId, false, false),
            new AccountMeta(associatedTokenProgramId, false, false),
            new AccountMeta(systemProgramId, false, false),
            new AccountMeta(rentSysvarId, false, false)
        );

        byte[] data = discriminator("participate_activity");
        return buildAndSend(List.of(new SolInstruction(programId, accounts, data)), List.of(privateKey));
    }

    /** set_merkle_root(rank winner's ClaimReward accounts, root: [u8;32]). */
    public String setMerkleRoot(byte[] authoritySigner, byte[] paymentMint, byte[] root, byte[] privateKey) {
        byte[] activity = deriveActivityAddress(authoritySigner);
        byte[] signerTokenAccount = findAssociatedTokenAddress(paymentMint, authoritySigner);
        byte[] poolTokenAccount = deriveActivityPool(authoritySigner);

        List<AccountMeta> accounts = List.of(
            new AccountMeta(authoritySigner, true, true),
            new AccountMeta(activity, false, true),
            // ClaimReward account order: pool before winner ATA.
            new AccountMeta(poolTokenAccount, false, true),
            new AccountMeta(signerTokenAccount, false, true),
            new AccountMeta(paymentMint, false, false),
            new AccountMeta(tokenProgramId, false, false),
            new AccountMeta(associatedTokenProgramId, false, false)
        );

        ByteArrayOutputStream dataBuf = new ByteArrayOutputStream();
        try {
            dataBuf.write(discriminator("set_merkle_root"));
            dataBuf.write(root);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return buildAndSend(List.of(new SolInstruction(programId, accounts, dataBuf.toByteArray())), List.of(privateKey));
    }

    public String claimReward(byte[] winner, byte[] authority, byte[] paymentMint, int rank, List<byte[]> proof, byte[] privateKey) {
        byte[] activity = deriveActivityAddress(authority);
        byte[] winnerTokenAccount = findAssociatedTokenAddress(paymentMint, winner);
        byte[] poolTokenAccount = deriveActivityPool(authority);

        // ClaimReward has exactly 7 accounts (no system_program/rent).
        // The pool PDA comes before the winner ATA, matching lib.rs:474.
        List<AccountMeta> accounts = List.of(
            new AccountMeta(winner, true, true),
            new AccountMeta(activity, false, true),
            new AccountMeta(poolTokenAccount, false, true),
            new AccountMeta(winnerTokenAccount, false, true),
            new AccountMeta(paymentMint, false, false),
            new AccountMeta(tokenProgramId, false, false),
            new AccountMeta(associatedTokenProgramId, false, false)
        );

        // claim_reward(rank: u8, proof: Vec<[u8;32]>)
        ByteArrayOutputStream dataBuf = new ByteArrayOutputStream();
        try {
            dataBuf.write(discriminator("claim_reward"));
            dataBuf.write(rank);
            dataBuf.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(proof.size()).array());
            for (byte[] p : proof) dataBuf.write(p);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        return buildAndSend(List.of(new SolInstruction(programId, accounts, dataBuf.toByteArray())), List.of(privateKey));
    }

    public String setUpline(byte[] signer, byte[] upline, byte[] target, byte[] privateKey) {
        byte[] commissionGraph = deriveCommissionGraph(target);

        List<AccountMeta> accounts = List.of(
            new AccountMeta(signer, true, true),
            new AccountMeta(upline, false, false),
            new AccountMeta(commissionGraph, false, true),
            new AccountMeta(target, false, false),
            new AccountMeta(systemProgramId, false, false),
            new AccountMeta(rentSysvarId, false, false)
        );

        byte[] data = discriminator("set_upline");
        return buildAndSend(List.of(new SolInstruction(programId, accounts, data)), List.of(privateKey));
    }

    public String distributeCommission(byte[] seller, byte[] paymentMint, long saleAmount, byte[] privateKey, byte[] uplineAta) {
        byte[] commissionGraph = deriveCommissionGraph(seller);
        byte[] sellerAta = findAssociatedTokenAddress(paymentMint, seller);

        List<AccountMeta> accounts = new ArrayList<>();
        accounts.add(new AccountMeta(seller, true, true));
        accounts.add(new AccountMeta(commissionGraph, false, true));
        accounts.add(new AccountMeta(sellerAta, false, true));
        // DistributeCommission order: upline ATA before payment mint / token program.
        accounts.add(new AccountMeta(uplineAta, false, true));
        accounts.add(new AccountMeta(paymentMint, false, false));
        accounts.add(new AccountMeta(tokenProgramId, false, false));

        ByteArrayOutputStream dataBuf = new ByteArrayOutputStream();
        try {
            dataBuf.write(discriminator("distribute_commission"));
            dataBuf.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(saleAmount).array());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        return buildAndSend(List.of(new SolInstruction(programId, accounts, dataBuf.toByteArray())), List.of(privateKey));
    }

    public String createCoupon(byte[] authority, byte[] mint, long discountAmount, long maxUses, byte[] merkleRoot, long expiry, byte[] privateKey) {
        byte[] coupon = deriveCouponAddress(authority);
        byte[] pool = deriveCouponPoolAddress(authority);

        List<AccountMeta> accounts = new ArrayList<>();
        accounts.add(new AccountMeta(authority, true, true));
        accounts.add(new AccountMeta(coupon, false, true));
        accounts.add(new AccountMeta(pool, false, true));
        accounts.add(new AccountMeta(mint, false, false));
        accounts.add(new AccountMeta(tokenProgramId, false, false));
        accounts.add(new AccountMeta(associatedTokenProgramId, false, false));
        accounts.add(new AccountMeta(systemProgramId, false, false));
        accounts.add(new AccountMeta(rentSysvarId, false, false));

        byte[] data;
        try {
            ByteArrayOutputStream buf = new ByteArrayOutputStream();
            buf.write(discriminator("create_coupon"));
            buf.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(discountAmount).array());
            buf.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(maxUses).array());
            buf.write(merkleRoot.length == 32 ? merkleRoot : new byte[32]);
            buf.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(expiry).array());
            data = buf.toByteArray();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        return buildAndSend(List.of(new SolInstruction(programId, accounts, data)), List.of(privateKey));
    }

    public String redeemCoupon(byte[] authority, byte[] user, byte[] mint, byte[] privateKey, List<byte[]> proof) {
        byte[] coupon = deriveCouponAddress(authority);
        byte[] pool = deriveCouponPoolAddress(authority);
        byte[] userAta = findAssociatedTokenAddress(mint, user);

        List<AccountMeta> accounts = new ArrayList<>();
        accounts.add(new AccountMeta(user, true, true));
        accounts.add(new AccountMeta(authority, false, false));
        accounts.add(new AccountMeta(coupon, false, true));
        accounts.add(new AccountMeta(pool, false, true));
        accounts.add(new AccountMeta(userAta, false, true));
        accounts.add(new AccountMeta(mint, false, false));
        accounts.add(new AccountMeta(tokenProgramId, false, false));
        accounts.add(new AccountMeta(associatedTokenProgramId, false, false));
        accounts.add(new AccountMeta(systemProgramId, false, false));

        ByteArrayOutputStream dataBuf = new ByteArrayOutputStream();
        try {
            dataBuf.write(discriminator("redeem_coupon"));
            dataBuf.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(proof.size()).array());
            for (byte[] p : proof) dataBuf.write(p);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        return buildAndSend(List.of(new SolInstruction(programId, accounts, dataBuf.toByteArray())), List.of(privateKey));
    }

    // ──────────────────────────────────────────────
    // Helpers
    // ──────────────────────────────────────────────

    /**
     * SPL associated token account = PDA of [owner, token_program, mint] under
     * the Associated Token Program. The old implementation hashed the seeds
     * without the bump / program id / PDA marker, so it did NOT produce a real ATA.
     */
    public byte[] findAssociatedTokenAddress(byte[] mint, byte[] owner) {
        return findProgramAddress(
            List.of(owner, tokenProgramId, mint),
            associatedTokenProgramId).address();
    }

    private static byte[] concat(byte[]... arrays) {
        try {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            for (byte[] arr : arrays) bos.write(arr);
            return bos.toByteArray();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static String base58Encode(byte[] input) {
        if (input.length == 0) return "";
        int zeros = 0;
        while (zeros < input.length && input[zeros] == 0) zeros++;
        byte[] copy = Arrays.copyOf(input, input.length);
        StringBuilder result = new StringBuilder();
        int length = 0;
        while (length < copy.length) {
            int remainder = 0;
            for (int i = length; i < copy.length; i++) {
                int digit = (copy[i] & 0xFF) + remainder * 256;
                copy[i] = (byte) (digit / 58);
                remainder = digit % 58;
            }
            result.insert(0, ALPHABET.charAt(remainder));
            while (length < copy.length && copy[length] == 0) length++;
        }
        while (zeros-- > 0) result.insert(0, '1');
        return result.toString();
    }

    public static byte[] decodeBase58(String input) {
        if (input.isEmpty()) return new byte[0];
        byte[] decoded = new byte[input.length() * 2];
        int length = 0;
        for (int i = 0; i < input.length(); i++) {
            int carry = BASE58_DECODE[input.charAt(i)];
            if (carry < 0) throw new IllegalArgumentException("Invalid base58 character: " + input.charAt(i));
            for (int j = 0; j < length; j++) {
                carry += (decoded[j] & 0xFF) * 58;
                decoded[j] = (byte) (carry & 0xFF);
                carry >>= 8;
            }
            while (carry > 0) {
                decoded[length++] = (byte) (carry & 0xFF);
                carry >>= 8;
            }
        }
        byte[] result = new byte[length];
        for (int i = 0; i < length; i++) result[i] = decoded[length - 1 - i];

        int zeros = 0;
        while (zeros < input.length() && input.charAt(zeros) == '1') zeros++;
        if (zeros > 0) {
            byte[] padded = new byte[zeros + result.length];
            System.arraycopy(result, 0, padded, zeros, result.length);
            return padded;
        }
        return result;
    }
}
