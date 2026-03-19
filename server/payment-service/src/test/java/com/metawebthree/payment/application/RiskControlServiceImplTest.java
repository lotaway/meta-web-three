package com.metawebthree.payment.application;

import com.metawebthree.common.generated.rpc.RiskScorerService;
import com.metawebthree.common.generated.rpc.ScoreRequest;
import com.metawebthree.common.generated.rpc.ScoreResponse;
import com.metawebthree.payment.infrastructure.persistence.mapper.ExchangeOrderRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;

import java.math.BigDecimal;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

import com.metawebthree.common.dto.UserRiskProfileDTO;
import com.metawebthree.common.rpc.UserRiskProfileService;

@ExtendWith(MockitoExtension.class)
class RiskControlServiceImplTest {

        @Mock
        private ExchangeOrderRepository exchangeOrderRepository;

        @Mock
        private RiskScorerService riskScorerService;

        @Mock
        private UserRiskProfileService userRiskProfileService;

        @InjectMocks
        private RiskControlServiceImpl riskControlService;

        @BeforeEach
        void setUp() {
                ReflectionTestUtils.setField(riskControlService, "minScore", 600);
                ReflectionTestUtils.setField(riskControlService, "singleLimitUSD", new BigDecimal("10000"));
                ReflectionTestUtils.setField(riskControlService, "dailyLimitUSD", new BigDecimal("50000"));
                ReflectionTestUtils.setField(riskControlService, "hourlyOrderLimit", 100);

                // We need to inject the mock into the private field annotated with
                // @DubboReference
                // @InjectMocks might miss it or we want to be explicit.
                // Actually @InjectMocks usually injects by type into fields.
                // But let's be safe.
                ReflectionTestUtils.setField(riskControlService, "riskScorerService", riskScorerService);
                ReflectionTestUtils.setField(riskControlService, "userRiskProfileService", userRiskProfileService);
        }

        @Test
        void testValidateOrder_HighRiskScore_ShouldThrowException() {
                // Given
                ScoreResponse lowScoreResponse = ScoreResponse.newBuilder()
                                .setScore(500) // Below 600
                                .setDecision("REJECT")
                                .build();
                when(riskScorerService.score(any(ScoreRequest.class))).thenReturn(lowScoreResponse);
                when(userRiskProfileService.getUserRiskProfile(any())).thenReturn(UserRiskProfileDTO.builder().build());

                // When/Then
                assertThrows(RuntimeException.class,
                                () -> riskControlService.validateOrder(1L, new BigDecimal("100"), "USD"));
        }

        @Test
        void testValidateOrder_GoodRiskScore_ShouldPass() {
                // Given
                ScoreResponse highScoreResponse = ScoreResponse.newBuilder()
                                .setScore(700) // Above 600
                                .setDecision("APPROVE")
                                .build();
                when(riskScorerService.score(any(ScoreRequest.class))).thenReturn(highScoreResponse);
                when(userRiskProfileService.getUserRiskProfile(any())).thenReturn(UserRiskProfileDTO.builder().build());

                // Also mock other validations to pass
                // validateSingleLimit, etc. depend on @Value fields which are set.
                // validateDailyLimit calls exchangeOrderRepository
                // validateFrequency calls exchangeOrderRepository
                // validateAbnormalBehavior calls exchangeOrderRepository

                // Mocking repo calls to return "safe" values
                when(exchangeOrderRepository.getTotalCompletedAmountByUserIdAndDateRange(any(), any()))
                                .thenReturn(BigDecimal.ZERO);
                when(exchangeOrderRepository.getCompletedOrderCountByUserIdAndDateRange(any(), any()))
                                .thenReturn(0L);
                when(exchangeOrderRepository.findByUserIdAndStatus(any(), any()))
                                .thenReturn(java.util.Collections.emptyList());

                // When/Then
                assertDoesNotThrow(() -> riskControlService.validateOrder(1L, new BigDecimal("100"), "USD"));
        }
}
