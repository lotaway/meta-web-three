package com.metawebthree.recommendation.infrastructure.persistence.repository;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.recommendation.domain.entity.RecommendationResult;
import com.metawebthree.recommendation.infrastructure.persistence.entity.RecommendationResultDO;
import com.metawebthree.recommendation.infrastructure.persistence.mapper.RecommendationResultMapper;

import java.time.LocalDateTime;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;

@ExtendWith(MockitoExtension.class)
class RecommendationResultRepositoryImplTest {

    @Mock
    private RecommendationResultMapper recommendationResultMapper;

    private RecommendationResultRepositoryImpl repository;

    @BeforeEach
    void setUp() {
        repository = new RecommendationResultRepositoryImpl(recommendationResultMapper);
    }

    @Test
    void save_shouldInsertNew() {
        RecommendationResult result = new RecommendationResult();
        result.setUserId(1L);
        result.setProductId(100L);
        result.setScore(85.0);
        result.setAlgorithm(RecommendationResult.RecommendationAlgorithm.POPULARITY);
        result.setIsClicked(false);
        result.setIsPurchased(false);

        when(recommendationResultMapper.insert(any(RecommendationResultDO.class)))
            .thenAnswer(invocation -> {
                ((RecommendationResultDO) invocation.getArgument(0)).setId(1L);
                return 1;
            });

        RecommendationResult saved = repository.save(result);

        assertNotNull(saved);
        assertEquals(1L, saved.getId());
        verify(recommendationResultMapper).insert(any(RecommendationResultDO.class));
    }

    @Test
    void save_shouldUpdateExisting() {
        RecommendationResult result = new RecommendationResult();
        result.setId(1L);
        result.setUserId(1L);
        result.setProductId(100L);
        result.setScore(90.0);

        when(recommendationResultMapper.updateById(any(RecommendationResultDO.class))).thenReturn(1);

        RecommendationResult saved = repository.save(result);

        assertNotNull(saved);
        assertEquals(1L, saved.getId());
        verify(recommendationResultMapper).updateById(any(RecommendationResultDO.class));
    }

    @Test
    void findByUserIdAndAlgorithm_shouldReturnResults() {
        RecommendationResultDO doEntity = new RecommendationResultDO();
        doEntity.setId(1L);
        doEntity.setUserId(1L);
        doEntity.setProductId(100L);
        doEntity.setScore(85.0);
        doEntity.setAlgorithm("POPULARITY");
        doEntity.setIsClicked(0);
        doEntity.setIsPurchased(0);

        when(recommendationResultMapper.selectList(any(LambdaQueryWrapper.class)))
            .thenReturn(List.of(doEntity));

        List<RecommendationResult> results = repository.findByUserIdAndAlgorithm(
            1L, RecommendationResult.RecommendationAlgorithm.POPULARITY, LocalDateTime.now());

        assertFalse(results.isEmpty());
        assertEquals(1, results.size());
        assertEquals(1L, results.get(0).getUserId());
        assertEquals(100L, results.get(0).getProductId());
        assertEquals(RecommendationResult.RecommendationAlgorithm.POPULARITY, results.get(0).getAlgorithm());
    }

    @Test
    void findByUserIdAndProductId_shouldReturnResults() {
        RecommendationResultDO doEntity = new RecommendationResultDO();
        doEntity.setId(1L);
        doEntity.setUserId(1L);
        doEntity.setProductId(100L);
        doEntity.setScore(85.0);
        doEntity.setAlgorithm("POPULARITY");
        doEntity.setIsClicked(0);
        doEntity.setIsPurchased(0);

        when(recommendationResultMapper.selectList(any(LambdaQueryWrapper.class)))
            .thenReturn(List.of(doEntity));

        List<RecommendationResult> results = repository.findByUserIdAndProductId(1L, 100L);

        assertFalse(results.isEmpty());
        assertEquals(1, results.size());
        assertEquals(100L, results.get(0).getProductId());
    }

    @Test
    void markAsClicked_shouldUpdateIsClicked() {
        RecommendationResultDO doEntity = new RecommendationResultDO();
        doEntity.setId(1L);
        doEntity.setIsClicked(0);

        when(recommendationResultMapper.selectById(1L)).thenReturn(doEntity);
        when(recommendationResultMapper.updateById(any())).thenReturn(1);

        repository.markAsClicked(1L);

        assertEquals(1, doEntity.getIsClicked());
        verify(recommendationResultMapper).updateById(doEntity);
    }

    @Test
    void markAsPurchased_shouldUpdateIsPurchased() {
        RecommendationResultDO doEntity = new RecommendationResultDO();
        doEntity.setId(1L);
        doEntity.setIsPurchased(0);

        when(recommendationResultMapper.selectById(1L)).thenReturn(doEntity);
        when(recommendationResultMapper.updateById(any())).thenReturn(1);

        repository.markAsPurchased(1L);

        assertEquals(1, doEntity.getIsPurchased());
        verify(recommendationResultMapper).updateById(doEntity);
    }

    @Test
    void deleteByExpiresAtBefore_shouldCallDelete() {
        when(recommendationResultMapper.delete(any(LambdaQueryWrapper.class))).thenReturn(5);

        repository.deleteByExpiresAtBefore(LocalDateTime.now());

        verify(recommendationResultMapper).delete(any(LambdaQueryWrapper.class));
    }

    @Test
    void countClicksByProductId_shouldReturnCount() {
        when(recommendationResultMapper.selectCount(any(LambdaQueryWrapper.class))).thenReturn(3L);

        Long count = repository.countClicksByProductId(100L);

        assertEquals(3L, count);
    }

    @Test
    void findByUserIdWithPagination_shouldReturnPage() {
        RecommendationResultDO doEntity = new RecommendationResultDO();
        doEntity.setId(1L);
        doEntity.setUserId(1L);
        doEntity.setProductId(100L);
        doEntity.setScore(85.0);
        doEntity.setAlgorithm("POPULARITY");

        when(recommendationResultMapper.selectList(any(LambdaQueryWrapper.class)))
            .thenReturn(List.of(doEntity, doEntity, doEntity));

        Pageable pageable = PageRequest.of(0, 2);
        List<RecommendationResult> results = repository.findByUserIdWithPagination(1L, LocalDateTime.now(), pageable);

        assertEquals(2, results.size());
    }

    @Test
    void toDomain_shouldMapIsClickedAndIsPurchased() {
        RecommendationResultDO doEntity = new RecommendationResultDO();
        doEntity.setId(1L);
        doEntity.setUserId(1L);
        doEntity.setProductId(100L);
        doEntity.setScore(85.0);
        doEntity.setAlgorithm("POPULARITY");
        doEntity.setIsClicked(1);
        doEntity.setIsPurchased(0);

        when(recommendationResultMapper.selectList(any(LambdaQueryWrapper.class)))
            .thenReturn(List.of(doEntity));

        List<RecommendationResult> results = repository.findByUserIdAndAlgorithm(
            1L, RecommendationResult.RecommendationAlgorithm.POPULARITY, LocalDateTime.now());

        assertTrue(results.get(0).getIsClicked());
        assertFalse(results.get(0).getIsPurchased());
    }
}
