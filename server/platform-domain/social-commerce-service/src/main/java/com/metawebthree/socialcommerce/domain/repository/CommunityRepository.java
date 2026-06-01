package com.metawebthree.socialcommerce.domain.repository;

import java.util.List;
import com.metawebthree.socialcommerce.domain.model.CommunityDO;

public interface CommunityRepository {
    void save(CommunityDO community);
    void update(CommunityDO community);
    CommunityDO findById(Long id);
    CommunityDO findByInviteCode(String inviteCode);
    List<CommunityDO> findByOwnerId(Long ownerId);
    List<CommunityDO> findByStatus(String status);
    void delete(Long id);
}