package com.metawebthree.cs.application;

import com.metawebthree.cs.domain.model.TransferLog;
import com.metawebthree.cs.infrastructure.persistence.mybatis.MybatisTransferLogMapper;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class TransferService {
    private final MybatisTransferLogMapper transferLogMapper;

    public TransferService(MybatisTransferLogMapper transferLogMapper) {
        this.transferLogMapper = transferLogMapper;
    }

    public TransferLog transfer(String sessionId, Long fromAgentId, Long toAgentId, String reason) {
        TransferLog transferLog = new TransferLog(sessionId, fromAgentId, toAgentId, reason);
        transferLogMapper.insert(transferLog);
        return transferLog;
    }

    public List<TransferLog> getTransferHistory(String sessionId) {
        return transferLogMapper.findBySessionId(sessionId);
    }
}