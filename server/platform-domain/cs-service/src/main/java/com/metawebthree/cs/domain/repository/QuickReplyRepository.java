package com.metawebthree.cs.domain.repository;

import com.metawebthree.cs.domain.model.QuickReply;

import java.util.List;
import java.util.Optional;

public interface QuickReplyRepository {
    QuickReply save(QuickReply quickReply);
    Optional<QuickReply> findById(Long id);
    List<QuickReply> findByGroupId(Long groupId);
    List<QuickReply> findAll();
    void deleteById(Long id);
}
