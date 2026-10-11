package com.metawebthree.cs.application;

import com.metawebthree.cs.domain.model.QuickReply;
import com.metawebthree.cs.domain.repository.QuickReplyRepository;

import java.util.List;

public class QuickReplyService {
    private final QuickReplyRepository quickReplyRepository;

    public QuickReplyService(QuickReplyRepository quickReplyRepository) {
        this.quickReplyRepository = quickReplyRepository;
    }

    public QuickReply create(Long groupId, String title, String content, String msgType) {
        QuickReply quickReply = new QuickReply();
        quickReply.setGroupId(groupId);
        quickReply.setTitle(title);
        quickReply.setContent(content);
        quickReply.setMsgType(msgType);
        return quickReplyRepository.save(quickReply);
    }

    public void delete(Long id) {
        quickReplyRepository.deleteById(id);
    }

    public List<QuickReply> listByGroup(Long groupId) {
        return quickReplyRepository.findByGroupId(groupId);
    }

    public List<QuickReply> listAll() {
        return quickReplyRepository.findAll();
    }
}
