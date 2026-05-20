package com.metawebthree.promotion.infrastructure.util;

import com.metawebthree.promotion.domain.ports.MerkleService;
import org.web3j.crypto.Hash;
import org.web3j.utils.Numeric;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.math.BigInteger;
import java.util.Arrays;

public class Web3MerkleService implements MerkleService {

    @Override
    public byte[] computeLeaf(String userAddress, String couponCode, long discount, long minPrice, long startTime, long endTime) {
        byte[] addrBytes = Numeric.hexStringToByteArray(userAddress);
        byte[] codeBytes = couponCode.getBytes(StandardCharsets.UTF_8);
        byte[] discountBytes = Numeric.toBytesPadded(BigInteger.valueOf(discount), 32);
        byte[] minPriceBytes = Numeric.toBytesPadded(BigInteger.valueOf(minPrice), 32);
        byte[] startTimeBytes = Numeric.toBytesPadded(BigInteger.valueOf(startTime), 32);
        byte[] endTimeBytes = Numeric.toBytesPadded(BigInteger.valueOf(endTime), 32);
        
        int totalLen = addrBytes.length + codeBytes.length + discountBytes.length + minPriceBytes.length + startTimeBytes.length + endTimeBytes.length;
        ByteBuffer buffer = ByteBuffer.allocate(totalLen);
        buffer.put(addrBytes).put(codeBytes).put(discountBytes).put(minPriceBytes).put(startTimeBytes).put(endTimeBytes);
        
        return Hash.sha3(buffer.array());
    }

    @Override
    public String getMerkleRoot(List<byte[]> leaves) {
        if (leaves == null || leaves.isEmpty()) return null;
        List<byte[]> currentLevel = sortLeaves(leaves);
        while (currentLevel.size() > 1) {
            currentLevel = buildNextLevel(currentLevel);
        }
        return Numeric.toHexString(currentLevel.get(0));
    }

    @Override
    public List<String> getMerkleProof(List<byte[]> leaves, byte[] leaf) {
        List<byte[]> currentLevel = sortLeaves(leaves);
        int index = findIndex(currentLevel, leaf);
        if (index == -1) return null;
        
        List<String> proof = new ArrayList<>();
        while (currentLevel.size() > 1) {
            int pairIndex = (index % 2 == 0) ? index + 1 : index - 1;
            if (pairIndex < currentLevel.size()) {
                proof.add(Numeric.toHexString(currentLevel.get(pairIndex)));
            }
            index = index / 2;
            currentLevel = buildNextLevel(currentLevel);
        }
        return proof;
    }

    private List<byte[]> buildNextLevel(List<byte[]> currentLevel) {
        List<byte[]> nextLevel = new ArrayList<>();
        for (int i = 0; i < currentLevel.size(); i += 2) {
            if (i + 1 < currentLevel.size()) {
                nextLevel.add(hashPair(currentLevel.get(i), currentLevel.get(i + 1)));
            } else {
                nextLevel.add(currentLevel.get(i));
            }
        }
        return nextLevel;
    }

    private int findIndex(List<byte[]> level, byte[] leaf) {
        for (int i = 0; i < level.size(); i++) {
            if (Arrays.equals(level.get(i), leaf)) return i;
        }
        return -1;
    }

    private byte[] hashPair(byte[] left, byte[] right) {
        BigInteger leftInt = Numeric.toBigInt(left);
        BigInteger rightInt = Numeric.toBigInt(right);
        return leftInt.compareTo(rightInt) < 0 ? Hash.sha3(concat(left, right)) : Hash.sha3(concat(right, left));
    }

    private List<byte[]> sortLeaves(List<byte[]> leaves) {
        List<byte[]> result = new ArrayList<>(leaves);
        Collections.sort(result, (a, b) -> Numeric.toBigInt(a).compareTo(Numeric.toBigInt(b)));
        return result;
    }

    private byte[] concat(byte[] a, byte[] b) {
        byte[] res = new byte[a.length + b.length];
        System.arraycopy(a, 0, res, 0, a.length);
        System.arraycopy(b, 0, res, a.length, b.length);
        return res;
    }
}
