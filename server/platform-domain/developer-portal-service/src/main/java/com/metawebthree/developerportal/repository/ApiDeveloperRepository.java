package com.metawebthree.developerportal.repository;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.developerportal.entity.ApiDeveloper;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;
import java.util.List;
import java.util.Optional;

@Mapper
public interface ApiDeveloperRepository extends BaseMapper<ApiDeveloper> {

    default Optional<ApiDeveloper> findByDeveloperId(String developerId) {
        return Optional.ofNullable(
            selectOne(new QueryWrapper<ApiDeveloper>().eq("developer_id", developerId)));
    }

    default Optional<ApiDeveloper> findByEmail(String email) {
        return Optional.ofNullable(
            selectOne(new QueryWrapper<ApiDeveloper>().eq("email", email)));
    }

    default List<ApiDeveloper> findByStatus(ApiDeveloper.DeveloperStatus status) {
        return selectList(new QueryWrapper<ApiDeveloper>().eq("status", status));
    }

    default boolean existsByEmail(String email) {
        return selectCount(new QueryWrapper<ApiDeveloper>().eq("email", email)) > 0;
    }

    default boolean existsByDeveloperId(String developerId) {
        return selectCount(new QueryWrapper<ApiDeveloper>().eq("developer_id", developerId)) > 0;
    }

    @Select("SELECT * FROM api_developer WHERE balance < #{thresholdCents} AND status = 'APPROVED'")
    List<ApiDeveloper> findByBalanceBelowThreshold(long thresholdCents);

    default void save(ApiDeveloper entity) {
        if (entity.getId() == null) {
            insert(entity);
        } else {
            updateById(entity);
        }
    }

    default void delete(ApiDeveloper entity) {
        deleteById(entity.getId());
    }
}
