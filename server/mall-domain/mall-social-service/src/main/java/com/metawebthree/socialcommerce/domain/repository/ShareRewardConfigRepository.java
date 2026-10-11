package com.metawebthree.socialcommerce.domain.repository;

import java.util.List;
import com.metawebthree.socialcommerce.domain.model.ShareRewardConfigDO;

public interface ShareRewardConfigRepository {
    void save(ShareRewardConfigDO config);
    void update(ShareRewardConfigDO config);
    ShareRewardConfigDO findById(Long id);
    ShareRewardConfigDO findActiveConfig();
    List<ShareRewardConfigDO> findByStatus(Integer status);
    void delete(Long id);
}