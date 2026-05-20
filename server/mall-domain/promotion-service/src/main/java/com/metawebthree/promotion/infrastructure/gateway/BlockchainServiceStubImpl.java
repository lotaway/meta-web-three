package com.metawebthree.promotion.infrastructure.gateway;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import com.metawebthree.promotion.domain.ports.BlockchainService;

@Service
public class BlockchainServiceStubImpl implements BlockchainService {
    private static final Logger log = LoggerFactory.getLogger(BlockchainServiceStubImpl.class);

    @Override
    public void setCouponBatchRoot(String batchId, String merkleRoot) {
        log.info("Stub blockchain publish for batchId={}, merkleRoot={}", batchId, merkleRoot);
    }
}
