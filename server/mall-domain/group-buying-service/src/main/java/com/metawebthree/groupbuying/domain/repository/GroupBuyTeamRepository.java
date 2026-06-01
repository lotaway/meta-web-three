package com.metawebthree.groupbuying.domain.repository;

import java.util.List;
import com.metawebthree.groupbuying.domain.model.GroupBuyTeamDO;

public interface GroupBuyTeamRepository {
    void save(GroupBuyTeamDO team);
    void update(GroupBuyTeamDO team);
    GroupBuyTeamDO findById(Long id);
    GroupBuyTeamDO findByTeamNo(String teamNo);
    List<GroupBuyTeamDO> findByLeaderId(Long leaderId);
    List<GroupBuyTeamDO> findByActivityId(Long activityId);
    List<GroupBuyTeamDO> findByStatus(String status);
    void delete(Long id);
}