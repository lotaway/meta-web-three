package com.metawebthree.developerportal.repository;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.developerportal.entity.OAuthApplication;
import org.apache.ibatis.annotations.Mapper;
import java.util.List;
import java.util.Optional;

@Mapper
public interface OAuthApplicationRepository extends BaseMapper<OAuthApplication> {

    default Optional<OAuthApplication> findByClientId(String clientId) {
        return Optional.ofNullable(
            selectOne(new QueryWrapper<OAuthApplication>().eq("client_id", clientId)));
    }

    default List<OAuthApplication> findByDeveloperId(String developerId) {
        return selectList(new QueryWrapper<OAuthApplication>().eq("developer_id", developerId));
    }

    default List<OAuthApplication> findByDeveloperIdAndStatus(
            String developerId, OAuthApplication.AppStatus status) {
        return selectList(new QueryWrapper<OAuthApplication>()
            .eq("developer_id", developerId).eq("status", status));
    }

    default boolean existsByClientId(String clientId) {
        return selectCount(new QueryWrapper<OAuthApplication>().eq("client_id", clientId)) > 0;
    }

    default void save(OAuthApplication entity) {
        if (entity.getId() == null) {
            insert(entity);
        } else {
            updateById(entity);
        }
    }

    default void delete(OAuthApplication entity) {
        deleteById(entity.getId());
    }
}
