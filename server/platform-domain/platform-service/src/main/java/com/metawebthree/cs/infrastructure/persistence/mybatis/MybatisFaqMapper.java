package com.metawebthree.cs.infrastructure.persistence.mybatis;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.cs.domain.model.Faq;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface MybatisFaqMapper extends BaseMapper<Faq> {

    @Select("SELECT * FROM cs_faq WHERE category = #{category} ORDER BY priority DESC, hit_count DESC")
    List<Faq> findByCategory(@Param("category") String category);

    @Select("SELECT * FROM cs_faq WHERE enabled = #{enabled} ORDER BY priority DESC, hit_count DESC")
    List<Faq> findByEnabled(@Param("enabled") Boolean enabled);

    @Select("SELECT * FROM cs_faq WHERE " +
            "question ILIKE CONCAT('%', #{keyword}, '%') OR " +
            "answer ILIKE CONCAT('%', #{keyword}, '%') OR " +
            "keywords ILIKE CONCAT('%', #{keyword}, '%') " +
            "ORDER BY hit_count DESC")
    List<Faq> searchByKeyword(@Param("keyword") String keyword);

    @Select("SELECT * FROM cs_faq ORDER BY relevance_score DESC, hit_count DESC LIMIT #{limit}")
    List<Faq> findTopByRelevance(@Param("limit") int limit);

    @Update("UPDATE cs_faq SET hit_count = hit_count + 1, update_time = NOW() WHERE id = #{id}")
    void incrementHitCount(@Param("id") Long id);

    @Update("UPDATE cs_faq SET relevance_score = #{score}, update_time = NOW() WHERE id = #{id}")
    void updateRelevanceScore(@Param("id") Long id, @Param("score") Double score);
}