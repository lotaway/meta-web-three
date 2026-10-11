package com.metawebthree.socialcommerce.infrastructure.persistence.repository;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import org.springframework.stereotype.Repository;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.socialcommerce.domain.model.CommunityDO;
import com.metawebthree.socialcommerce.domain.repository.CommunityRepository;

/**
 * 社区仓储（本地开发用内存实现）。
 */
@Repository
public class CommunityRepositoryImpl implements CommunityRepository {

    private final Map<Long, CommunityDO> store = new ConcurrentHashMap<>();

    @Override
    public void save(CommunityDO community) {
        if (community.getId() == null) {
            community.setId(IdWorker.getId());
        }
        store.put(community.getId(), community);
    }

    @Override
    public void update(CommunityDO community) {
        if (community.getId() != null) {
            store.put(community.getId(), community);
        }
    }

    @Override
    public CommunityDO findById(Long id) {
        return store.get(id);
    }

    @Override
    public CommunityDO findByInviteCode(String inviteCode) {
        return store.values().stream()
                .filter(c -> Objects.equals(c.getInviteCode(), inviteCode))
                .findFirst()
                .orElse(null);
    }

    @Override
    public List<CommunityDO> findByOwnerId(Long ownerId) {
        return store.values().stream()
                .filter(c -> Objects.equals(c.getOwnerId(), ownerId))
                .collect(Collectors.toList());
    }

    @Override
    public List<CommunityDO> findByStatus(String status) {
        return store.values().stream()
                .filter(c -> Objects.equals(c.getStatus(), status))
                .collect(Collectors.toList());
    }

    @Override
    public void delete(Long id) {
        store.remove(id);
    }
}
