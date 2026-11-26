package com.metawebthree.wallet.infrastructure.solana;

import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

class KeccakMerkleTest {

    @Test
    void hash_matchesPublishedKeccak256Vectors() {
        // Published Keccak-256 vectors (Ethereum flavour). Java's SHA3-256 has
        // different padding and would produce different values.
        assertEquals(
            "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470",
            hex(KeccakMerkle.hash(new byte[0])));
        assertEquals(
            "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45",
            hex(KeccakMerkle.hash("abc".getBytes(StandardCharsets.UTF_8))));
    }

    @Test
    void leaf_isKeccakOfUserPubkey() {
        byte[] user = "abc".getBytes(StandardCharsets.UTF_8);
        assertArrayEquals(KeccakMerkle.hash(user), KeccakMerkle.leaf(user));
    }

    @Test
    void root_walksProofWithSortedPair() {
        byte[] alice = KeccakMerkle.leaf("alice".getBytes(StandardCharsets.UTF_8));
        byte[] bob = KeccakMerkle.leaf("bob".getBytes(StandardCharsets.UTF_8));

        byte[] combined = lexicallySortedConcat(alice, bob);
        byte[] expectedRoot = KeccakMerkle.hash(combined);

        // The on-chain proof walk is symmetric: either leaf + the other as proof
        // must resolve to the same root because the pair is sorted before hashing.
        assertArrayEquals(expectedRoot, KeccakMerkle.root(alice, List.of(bob)));
        assertArrayEquals(expectedRoot, KeccakMerkle.root(bob, List.of(alice)));
    }

    @Test
    void root_singleLeafProof_isLeafItself() {
        byte[] leaf = KeccakMerkle.leaf("alice".getBytes(StandardCharsets.UTF_8));
        assertArrayEquals(leaf, KeccakMerkle.root(leaf, List.of()));
    }

    @Test
    void root_wrongProofDiffers() {
        byte[] alice = KeccakMerkle.leaf("alice".getBytes(StandardCharsets.UTF_8));
        byte[] bob = KeccakMerkle.leaf("bob".getBytes(StandardCharsets.UTF_8));
        byte[] carol = KeccakMerkle.leaf("carol".getBytes(StandardCharsets.UTF_8));

        byte[] correct = KeccakMerkle.root(alice, List.of(bob));
        byte[] wrong = KeccakMerkle.root(alice, List.of(carol));
        assertFalse(Arrays.equals(correct, wrong));
    }

    @Test
    void root_threeLevels_isStable() {
        byte[] a = KeccakMerkle.leaf("a".getBytes(StandardCharsets.UTF_8));
        byte[] b = KeccakMerkle.leaf("b".getBytes(StandardCharsets.UTF_8));
        byte[] c = KeccakMerkle.leaf("c".getBytes(StandardCharsets.UTF_8));

        List<byte[]> proof = new ArrayList<>(List.of(b, c));
        byte[] root = KeccakMerkle.root(a, proof);
        assertEquals(32, root.length);
        assertNotEquals(hex(a), hex(root));
        // Recomputing with the same inputs must be deterministic.
        assertArrayEquals(root, KeccakMerkle.root(a, List.of(b, c)));
    }

    private static byte[] lexicallySortedConcat(byte[] a, byte[] b) {
        byte[] combined = new byte[64];
        if (compareUnsigned(a, b) <= 0) {
            System.arraycopy(a, 0, combined, 0, 32);
            System.arraycopy(b, 0, combined, 32, 32);
        } else {
            System.arraycopy(b, 0, combined, 0, 32);
            System.arraycopy(a, 0, combined, 32, 32);
        }
        return combined;
    }

    private static int compareUnsigned(byte[] a, byte[] b) {
        for (int i = 0; i < Math.min(a.length, b.length); i++) {
            int cmp = Byte.compareUnsigned(a[i], b[i]);
            if (cmp != 0) return cmp;
        }
        return Integer.compare(a.length, b.length);
    }

    private static String hex(byte[] bytes) {
        StringBuilder sb = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) sb.append(String.format("%02x", b & 0xff));
        return sb.toString();
    }
}
