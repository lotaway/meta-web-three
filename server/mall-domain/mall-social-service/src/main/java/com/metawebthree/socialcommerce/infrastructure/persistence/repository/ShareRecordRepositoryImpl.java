package com.metawebthree.socialcommerce.infrastructure.persistence.repository;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import org.springframework.stereotype.Repository;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.socialcommerce.domain.model.ShareRecordDO;
import com.metawebthree.socialcommerce.domain.repository.ShareRecordRepository;

/**
 * 分享记录仓储（本地开发用内存实现）。
 */
@Repository
public class ShareRecordRepositoryImpl implements ShareRecordRepository {

    private final Map<Long, ShareRecordDO> store = new ConcurrentHashMap<>();

    @Override
    public void save(ShareRecordDO record) {
        if (record.getId() == null) {
            record.setId(IdWorker.getId());
        }
        store.put(record.getId(), record);
    }

    @Override
    public void update(ShareRecordDO record) {
        if (record.getId() != null) {
            store.put(record.getId(), record);
        }
    }

    @Override
    public ShareRecordDO findById(Long id) {
        return store.get(id);
    }

    @Override
    public List<ShareRecordDO> findBySharerId(Long sharerId) {
        return store.values().stream()
                .filter(r -> Objects.equals(r.getSharerId(), sharerId))
                .collect(Collectors.toList());
    }

    @Override
    public List<ShareRecordDO> findBySharedItemIdAndType(Long itemId, String itemType) {
        return store.values().stream()
                .filter(r -> Objects.equals(r.getSharedItemId(), itemId) && Objects.equals(r.getItemType(), itemType))
                .collect(Collectors.toList());
    }

    @Override
    public ShareRecordDO findByShareUrl(String shareUrl) {
        return store.values().stream()
                .filter(r -> Objects.equals(r.getShareUrl(), shareUrl))
                .findFirst()
                .orElse(null);
    }

    @Override
    public void delete(Long id) {
        store.remove(id);
    }
}
