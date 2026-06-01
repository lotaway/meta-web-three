package com.metawebthree.socialcommerce.domain.repository;

import java.util.List;
import com.metawebthree.socialcommerce.domain.model.ShareRecordDO;

public interface ShareRecordRepository {
    void save(ShareRecordDO record);
    void update(ShareRecordDO record);
    ShareRecordDO findById(Long id);
    List<ShareRecordDO> findBySharerId(Long sharerId);
    List<ShareRecordDO> findBySharedItemIdAndType(Long itemId, String itemType);
    ShareRecordDO findByShareUrl(String shareUrl);
    void delete(Long id);
}