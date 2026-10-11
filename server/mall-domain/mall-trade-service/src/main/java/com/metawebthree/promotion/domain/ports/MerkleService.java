package com.metawebthree.promotion.domain.ports;

import java.util.List;

public interface MerkleService {
    byte[] computeLeaf(String userAddress, String couponCode, long discount, long minPrice, long startTime, long endTime);
    String getMerkleRoot(List<byte[]> leaves);
    List<String> getMerkleProof(List<byte[]> leaves, byte[] leaf);
}
