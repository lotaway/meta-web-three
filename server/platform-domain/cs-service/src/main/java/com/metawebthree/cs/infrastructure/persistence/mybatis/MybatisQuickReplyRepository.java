package com.metawebthree.cs.infrastructure.persistence.mybatis;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.cs.domain.model.QuickReply;
import com.metawebthree.cs.domain.repository.QuickReplyRepository;

import java.util.List;
import java.util.Optional;

public class MybatisQuickReplyRepository implements QuickReplyRepository {
    private final MybatisQuickReplyMapper mapper;

    public MybatisQuickReplyRepository(MybatisQuickReplyMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public QuickReply save(QuickReply quickReply) {
        if (quickReply.getId() == null) {
            mapper.insert(quickReply);
        } else {
            mapper.updateById(quickReply);
        }
        return quickReply;
    }

    @Override
    public Optional<QuickReply> findById(Long id) {
        return Optional.ofNullable(mapper.selectById(id));
    }

    @Override
    public List<QuickReply> findByGroupId(Long groupId) {
        LambdaQueryWrapper<QuickReply> query = new LambdaQueryWrapper<QuickReply>()
                .eq(QuickReply::getGroupId, groupId)
                .orderByAsc(QuickReply::getSort);
        return mapper.selectList(query);
    }

    @Override
    public List<QuickReply> findAll() {
        return mapper.selectList(null);
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }
}
