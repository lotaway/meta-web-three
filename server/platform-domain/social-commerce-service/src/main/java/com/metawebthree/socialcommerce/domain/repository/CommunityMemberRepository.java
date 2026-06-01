package com.metawebthree.socialcommerce.domain.repository;

import java.util.List;
import com.metawebthree.socialcommerce.domain.model.CommunityMemberDO;

public interface CommunityMemberRepository {
    void save(CommunityMemberDO member);
    void update(CommunityMemberDO member);
    CommunityMemberDO findById(Long id);
    CommunityMemberDO findByCommunityIdAndUserId(Long communityId, Long userId);
    List<CommunityMemberDO> findByCommunityId(Long communityId);
    List<CommunityMemberDO> findByUserId(Long userId);
    List<CommunityMemberDO> findByRole(String role);
    Boolean existsByCommunityIdAndUserId(Long communityId, Long userId);
    void delete(Long id);
}