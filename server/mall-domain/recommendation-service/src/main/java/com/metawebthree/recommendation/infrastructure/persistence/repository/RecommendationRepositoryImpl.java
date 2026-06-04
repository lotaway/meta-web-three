package com.metawebthree.recommendation.infrastructure.persistence.repository;

import com.metawebthree.recommendation.domain.entity.Recommendation;
import com.metawebthree.recommendation.domain.repository.RecommendationRepository;
import com.metawebthree.recommendation.infrastructure.persistence.entity.RecommendationDO;
import com.metawebthree.recommendation.infrastructure.persistence.mapper.RecommendationMapper;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import org.springframework.stereotype.Repository;

@Repository
public class RecommendationRepositoryImpl implements RecommendationRepository {

    private final RecommendationMapper recommendationMapper;

    public RecommendationRepositoryImpl(RecommendationMapper recommendationMapper) {
        this.recommendationMapper = recommendationMapper;
    }

    @Override
    public Optional<Recommendation> findById(Long id) {
        return Optional.ofNullable(recommendationMapper.selectById(id))
            .map(this::toDomain);
    }

    @Override
    public List<Recommendation> findByUserId(Long userId) {
        LambdaQueryWrapper<RecommendationDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(RecommendationDO::getUserId, userId);
        return recommendationMapper.selectList(wrapper).stream()
            .map(this::toDomain)
            .collect(Collectors.toList());
    }

    @Override
    public List<Recommendation> findByUserIdAndScene(Long userId, String scene) {
        LambdaQueryWrapper<RecommendationDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(RecommendationDO::getUserId, userId)
            .eq(RecommendationDO::getScene, scene);
        return recommendationMapper.selectList(wrapper).stream()
            .map(this::toDomain)
            .collect(Collectors.toList());
    }

    @Override
    public List<Recommendation> findByStatus(Recommendation.RecommendationStatus status) {
        LambdaQueryWrapper<RecommendationDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(RecommendationDO::getStatus, status.name());
        return recommendationMapper.selectList(wrapper).stream()
            .map(this::toDomain)
            .collect(Collectors.toList());
    }

    @Override
    public Recommendation save(Recommendation recommendation) {
        RecommendationDO recommendationDO = toDO(recommendation);
        if (recommendation.getId() == null) {
            recommendationMapper.insert(recommendationDO);
            recommendation.setId(recommendationDO.getId());
        } else {
            recommendationMapper.updateById(recommendationDO);
        }
        return recommendation;
    }

    @Override
    public void update(Recommendation recommendation) {
        recommendationMapper.updateById(toDO(recommendation));
    }

    @Override
    public void deleteById(Long id) {
        recommendationMapper.deleteById(id);
    }

    @Override
    public List<Recommendation> findAll() {
        return recommendationMapper.selectList(null).stream()
            .map(this::toDomain)
            .collect(Collectors.toList());
    }

    @Override
    public long count() {
        return recommendationMapper.selectCount(null);
    }

    @Override
    public void recordBehavior(Long userId, String skuCode, String behaviorType) {
        LambdaQueryWrapper<RecommendationDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(RecommendationDO::getUserId, userId);
        List<RecommendationDO> recs = recommendationMapper.selectList(wrapper);
        for (RecommendationDO rec : recs) {
            if ("CLICK".equals(behaviorType)) {
                rec.setClickCount(rec.getClickCount() != null ? rec.getClickCount() + 1 : 1);
            } else if ("CONVERSION".equals(behaviorType)) {
                rec.setConversionCount(rec.getConversionCount() != null ? rec.getConversionCount() + 1 : 1);
            } else if ("IMPRESSION".equals(behaviorType)) {
                rec.setImpressionCount(rec.getImpressionCount() != null ? rec.getImpressionCount() + 1 : 1);
            }
            recommendationMapper.updateById(rec);
        }
    }

    private Recommendation toDomain(RecommendationDO recommendationDO) {
        Recommendation recommendation = new Recommendation();
        recommendation.setId(recommendationDO.getId());
        recommendation.setUserId(recommendationDO.getUserId());
        recommendation.setScene(recommendationDO.getScene());
        recommendation.setAlgorithm(Recommendation.RecommendationAlgorithm.valueOf(recommendationDO.getAlgorithm()));
        recommendation.setScore(recommendationDO.getScore());
        recommendation.setStatus(Recommendation.RecommendationStatus.valueOf(recommendationDO.getStatus()));
        recommendation.setExpiresAt(recommendationDO.getExpiresAt());
        recommendation.setClickCount(recommendationDO.getClickCount());
        recommendation.setConversionCount(recommendationDO.getConversionCount());
        recommendation.setImpressionCount(recommendationDO.getImpressionCount());
        return recommendation;
    }

    private RecommendationDO toDO(Recommendation recommendation) {
        RecommendationDO recommendationDO = new RecommendationDO();
        recommendationDO.setId(recommendation.getId());
        recommendationDO.setUserId(recommendation.getUserId());
        recommendationDO.setScene(recommendation.getScene());
        recommendationDO.setAlgorithm(recommendation.getAlgorithm().name());
        recommendationDO.setScore(recommendation.getScore());
        recommendationDO.setStatus(recommendation.getStatus().name());
        recommendationDO.setCreatedAt(recommendation.getCreatedAt());
        recommendationDO.setExpiresAt(recommendation.getExpiresAt());
        recommendationDO.setClickCount(recommendation.getClickCount());
        recommendationDO.setConversionCount(recommendation.getConversionCount());
        recommendationDO.setImpressionCount(recommendation.getImpressionCount());
        return recommendationDO;
    }
}
