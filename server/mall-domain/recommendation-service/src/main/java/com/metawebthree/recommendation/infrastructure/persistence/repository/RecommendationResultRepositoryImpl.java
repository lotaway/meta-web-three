package com.metawebthree.recommendation.infrastructure.persistence.repository;

import com.metawebthree.recommendation.domain.entity.RecommendationResult;
import com.metawebthree.recommendation.domain.repository.RecommendationResultRepository;
import com.metawebthree.recommendation.infrastructure.persistence.entity.RecommendationResultDO;
import com.metawebthree.recommendation.infrastructure.persistence.mapper.RecommendationResultMapper;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Repository;

@Repository
public class RecommendationResultRepositoryImpl implements RecommendationResultRepository {

    private final RecommendationResultMapper recommendationResultMapper;

    public RecommendationResultRepositoryImpl(RecommendationResultMapper recommendationResultMapper) {
        this.recommendationResultMapper = recommendationResultMapper;
    }

    @Override
    public RecommendationResult save(RecommendationResult recommendationResult) {
        RecommendationResultDO recommendationResultDO = toDO(recommendationResult);
        if (recommendationResult.getId() == null) {
            recommendationResultMapper.insert(recommendationResultDO);
            recommendationResult.setId(recommendationResultDO.getId());
        } else {
            recommendationResultMapper.updateById(recommendationResultDO);
        }
        return recommendationResult;
    }

    @Override
    public List<RecommendationResult> findByUserIdOrderByScoreDesc(Long userId, LocalDateTime now) {
        LambdaQueryWrapper<RecommendationResultDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(RecommendationResultDO::getUserId, userId)
            .gt(RecommendationResultDO::getExpiresAt, now)
            .orderByDesc(RecommendationResultDO::getScore);
        return recommendationResultMapper.selectList(wrapper).stream()
            .map(this::toDomain)
            .collect(Collectors.toList());
    }

    @Override
    public List<RecommendationResult> findByUserIdAndAlgorithm(Long userId, RecommendationResult.RecommendationAlgorithm algorithm, LocalDateTime now) {
        LambdaQueryWrapper<RecommendationResultDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(RecommendationResultDO::getUserId, userId)
            .eq(RecommendationResultDO::getAlgorithm, algorithm.name())
            .gt(RecommendationResultDO::getExpiresAt, now)
            .orderByDesc(RecommendationResultDO::getScore);
        return recommendationResultMapper.selectList(wrapper).stream()
            .map(this::toDomain)
            .collect(Collectors.toList());
    }

    @Override
    public List<RecommendationResult> findByUserIdWithPagination(Long userId, LocalDateTime now, Pageable pageable) {
        LambdaQueryWrapper<RecommendationResultDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(RecommendationResultDO::getUserId, userId)
            .gt(RecommendationResultDO::getExpiresAt, now)
            .orderByDesc(RecommendationResultDO::getScore);
        List<RecommendationResultDO> allList = recommendationResultMapper.selectList(wrapper);
        int offset = (int) pageable.getOffset();
        int pageSize = pageable.getPageSize();
        int end = Math.min(offset + pageSize, allList.size());
        if (offset >= allList.size()) {
            return List.of();
        }
        return allList.subList(offset, end).stream()
            .map(this::toDomain)
            .collect(Collectors.toList());
    }

    @Override
    public void deleteByExpiresAtBefore(LocalDateTime timestamp) {
        LambdaQueryWrapper<RecommendationResultDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.lt(RecommendationResultDO::getExpiresAt, timestamp);
        recommendationResultMapper.delete(wrapper);
    }

    @Override
    public void markAsClicked(Long id) {
        RecommendationResultDO recommendationResultDO = recommendationResultMapper.selectById(id);
        if (recommendationResultDO != null) {
            recommendationResultDO.setIsClicked(1);
            recommendationResultMapper.updateById(recommendationResultDO);
        }
    }

    @Override
    public void markAsPurchased(Long id) {
        RecommendationResultDO recommendationResultDO = recommendationResultMapper.selectById(id);
        if (recommendationResultDO != null) {
            recommendationResultDO.setIsPurchased(1);
            recommendationResultMapper.updateById(recommendationResultDO);
        }
    }

    @Override
    public Long countClicksByProductId(Long productId) {
        LambdaQueryWrapper<RecommendationResultDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(RecommendationResultDO::getProductId, productId)
            .eq(RecommendationResultDO::getIsClicked, 1);
        return recommendationResultMapper.selectCount(wrapper);
    }

    @Override
    public Long countPurchasesByProductId(Long productId) {
        LambdaQueryWrapper<RecommendationResultDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(RecommendationResultDO::getProductId, productId)
            .eq(RecommendationResultDO::getIsPurchased, 1);
        return recommendationResultMapper.selectCount(wrapper);
    }

    @Override
    public List<RecommendationResult> findAll() {
        return recommendationResultMapper.selectList(new LambdaQueryWrapper<>()).stream()
            .map(this::toDomain)
            .collect(Collectors.toList());
    }

    @Override
    public void deleteById(Long id) {
        recommendationResultMapper.deleteById(id);
    }

    private RecommendationResult toDomain(RecommendationResultDO recommendationResultDO) {
        RecommendationResult recommendationResult = new RecommendationResult();
        recommendationResult.setId(recommendationResultDO.getId());
        recommendationResult.setUserId(recommendationResultDO.getUserId());
        recommendationResult.setProductId(recommendationResultDO.getProductId());
        recommendationResult.setScore(recommendationResultDO.getScore());
        recommendationResult.setAlgorithm(RecommendationResult.RecommendationAlgorithm.valueOf(recommendationResultDO.getAlgorithm()));
        recommendationResult.setReason(recommendationResultDO.getReason());
        recommendationResult.setPosition(recommendationResultDO.getPosition());
        recommendationResult.setCreatedAt(recommendationResultDO.getCreatedAt());
        recommendationResult.setExpiresAt(recommendationResultDO.getExpiresAt());
        recommendationResult.setIsClicked(recommendationResultDO.getIsClicked() != null && recommendationResultDO.getIsClicked() == 1);
        recommendationResult.setIsPurchased(recommendationResultDO.getIsPurchased() != null && recommendationResultDO.getIsPurchased() == 1);
        return recommendationResult;
    }

    private RecommendationResultDO toDO(RecommendationResult recommendationResult) {
        RecommendationResultDO recommendationResultDO = new RecommendationResultDO();
        recommendationResultDO.setId(recommendationResult.getId());
        recommendationResultDO.setUserId(recommendationResult.getUserId());
        recommendationResultDO.setProductId(recommendationResult.getProductId());
        recommendationResultDO.setScore(recommendationResult.getScore());
        recommendationResultDO.setAlgorithm(recommendationResult.getAlgorithm().name());
        recommendationResultDO.setReason(recommendationResult.getReason());
        recommendationResultDO.setPosition(recommendationResult.getPosition());
        recommendationResultDO.setCreatedAt(recommendationResult.getCreatedAt());
        recommendationResultDO.setExpiresAt(recommendationResult.getExpiresAt());
        recommendationResultDO.setIsClicked(recommendationResult.getIsClicked() != null && recommendationResult.getIsClicked() ? 1 : 0);
        recommendationResultDO.setIsPurchased(recommendationResult.getIsPurchased() != null && recommendationResult.getIsPurchased() ? 1 : 0);
        return recommendationResultDO;
    }
}
