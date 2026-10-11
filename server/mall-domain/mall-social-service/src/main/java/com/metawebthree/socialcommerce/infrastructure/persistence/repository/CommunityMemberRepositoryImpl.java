package com.metawebthree.socialcommerce.infrastructure.persistence.repository;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import org.springframework.stereotype.Repository;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.socialcommerce.domain.model.CommunityMemberDO;
import com.metawebthree.socialcommerce.domain.repository.CommunityMemberRepository;

/**
 * 社区成员仓储（本地开发用内存实现）。
 */
@Repository
public class CommunityMemberRepositoryImpl implements CommunityMemberRepository {

    private final Map<Long, CommunityMemberDO> store = new ConcurrentHashMap<>();

    @Override
    public void save(CommunityMemberDO member) {
        if (member.getId() == null) {
            member.setId(IdWorker.getId());
        }
        store.put(member.getId(), member);
    }

    @Override
    public void update(CommunityMemberDO member) {
        if (member.getId() != null) {
            store.put(member.getId(), member);
        }
    }

    @Override
    public CommunityMemberDO findById(Long id) {
        return store.get(id);
    }

    @Override
    public CommunityMemberDO findByCommunityIdAndUserId(Long communityId, Long userId) {
        return store.values().stream()
                .filter(m -> Objects.equals(m.getCommunityId(), communityId) && Objects.equals(m.getUserId(), userId))
                .findFirst()
                .orElse(null);
    }

    @Override
    public List<CommunityMemberDO> findByCommunityId(Long communityId) {
        return store.values().stream()
                .filter(m -> Objects.equals(m.getCommunityId(), communityId))
                .collect(Collectors.toList());
    }

    @Override
    public List<CommunityMemberDO> findByUserId(Long userId) {
        return store.values().stream()
                .filter(m -> Objects.equals(m.getUserId(), userId))
                .collect(Collectors.toList());
    }

    @Override
    public List<CommunityMemberDO> findByRole(String role) {
        return store.values().stream()
                .filter(m -> Objects.equals(m.getRole(), role))
                .collect(Collectors.toList());
    }

    @Override
    public Boolean existsByCommunityIdAndUserId(Long communityId, Long userId) {
        return store.values().stream()
                .anyMatch(m -> Objects.equals(m.getCommunityId(), communityId) && Objects.equals(m.getUserId(), userId));
    }

    @Override
    public void delete(Long id) {
        store.remove(id);
    }
}
