package com.metawebthree.developerportal.repository;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.developerportal.entity.ApiKey;
import org.apache.ibatis.annotations.Mapper;
import java.util.List;
import java.util.Optional;

@Mapper
public interface ApiKeyRepository extends BaseMapper<ApiKey> {

    default Optional<ApiKey> findByKeyId(String keyId) {
        return Optional.ofNullable(
            selectOne(new QueryWrapper<ApiKey>().eq("key_id", keyId)));
    }

    default List<ApiKey> findByDeveloperId(String developerId) {
        return selectList(new QueryWrapper<ApiKey>().eq("developer_id", developerId));
    }

    default List<ApiKey> findByDeveloperIdAndStatus(String developerId, ApiKey.KeyStatus status) {
        return selectList(new QueryWrapper<ApiKey>()
            .eq("developer_id", developerId).eq("status", status));
    }

    default boolean existsByKeyId(String keyId) {
        return selectCount(new QueryWrapper<ApiKey>().eq("key_id", keyId)) > 0;
    }

    default List<ApiKey> findByStatus(ApiKey.KeyStatus status) {
        return selectList(new QueryWrapper<ApiKey>().eq("status", status));
    }

    default void save(ApiKey entity) {
        if (entity.getId() == null) {
            insert(entity);
        } else {
            updateById(entity);
        }
    }
}
