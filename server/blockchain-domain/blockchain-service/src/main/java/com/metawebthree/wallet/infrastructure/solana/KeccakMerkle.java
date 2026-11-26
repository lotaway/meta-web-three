package com.metawebthree.wallet.infrastructure.solana;

import org.bouncycastle.crypto.digests.KeccakDigest;

import java.util.Arrays;
import java.util.List;

/**
 * Keccak-256 Merkle helpers that mirror contracts/solana/lib.rs::keccak_hashv
 * and redeem_coupon. Java's MessageDigest "SHA3-256" is NOT Keccak-256 (different
 * padding), so Bouncy Castle's KeccakDigest is used explicitly.
 */
public final class KeccakMerkle {

    private KeccakMerkle() {
    }

    public static byte[] hash(byte[]... values) {
        KeccakDigest digest = new KeccakDigest(256);
        for (byte[] value : values) {
            digest.update(value, 0, value.length);
        }
        byte[] out = new byte[32];
        digest.doFinal(out, 0);
        return out;
    }

    public static byte[] leaf(byte[] userPubkey) {
        return hash(userPubkey);
    }

    /** Mirrors the on-chain proof walk: sort the pair, concat, keccak. */
    public static byte[] root(byte[] leaf, List<byte[]> proof) {
        byte[] computed = leaf;
        for (byte[] element : proof) {
            byte[] combined = new byte[64];
            if (lexicographicLessOrEqual(computed, element)) {
                System.arraycopy(computed, 0, combined, 0, 32);
                System.arraycopy(element, 0, combined, 32, 32);
            } else {
                System.arraycopy(element, 0, combined, 0, 32);
                System.arraycopy(computed, 0, combined, 32, 32);
            }
            computed = hash(combined);
        }
        return computed;
    }

    private static boolean lexicographicLessOrEqual(byte[] a, byte[] b) {
        for (int i = 0; i < Math.min(a.length, b.length); i++) {
            int cmp = Byte.compareUnsigned(a[i], b[i]);
            if (cmp != 0) return cmp < 0;
        }
        return a.length <= b.length;
    }
}
