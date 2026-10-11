package com.metawebthree.developerportal.repository;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.developerportal.entity.ApiSubscription;
import org.apache.ibatis.annotations.Mapper;
import java.util.List;
import java.util.Optional;

@Mapper
public interface ApiSubscriptionRepository extends BaseMapper<ApiSubscription> {

    default Optional<ApiSubscription> findBySubscriptionId(String subscriptionId) {
        return Optional.ofNullable(
            selectOne(new QueryWrapper<ApiSubscription>().eq("subscription_id", subscriptionId)));
    }

    default List<ApiSubscription> findByDeveloperId(String developerId) {
        return selectList(new QueryWrapper<ApiSubscription>().eq("developer_id", developerId));
    }

    default List<ApiSubscription> findByDeveloperIdAndStatus(
            String developerId, ApiSubscription.SubscriptionStatus status) {
        return selectList(new QueryWrapper<ApiSubscription>()
            .eq("developer_id", developerId).eq("status", status));
    }

    default List<ApiSubscription> findByStatus(ApiSubscription.SubscriptionStatus status) {
        return selectList(new QueryWrapper<ApiSubscription>().eq("status", status));
    }

    default boolean existsByDeveloperIdAndApiPattern(String developerId, String apiPattern) {
        return selectCount(new QueryWrapper<ApiSubscription>()
            .eq("developer_id", developerId).eq("api_pattern", apiPattern)) > 0;
    }

    default void save(ApiSubscription entity) {
        if (entity.getId() == null) {
            insert(entity);
        } else {
            updateById(entity);
        }
    }
}
