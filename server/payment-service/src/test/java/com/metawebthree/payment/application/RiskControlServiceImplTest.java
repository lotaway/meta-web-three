package com.metawebthree.payment.application;

import com.metawebthree.common.generated.rpc.RiskScorerService;
import com.metawebthree.common.generated.rpc.ScoreRequest;
import com.metawebthree.common.generated.rpc.ScoreResponse;
import com.metawebthree.common.generated.rpc.UserRiskProfileService;
import com.metawebthree.common.generated.rpc.UserRiskProfile;
import com.metawebthree.common.generated.rpc.GetUserRiskProfileRequest;
import com.metawebthree.common.generated.rpc.GetUserRiskProfileResponse;
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

                ReflectionTestUtils.setField(riskControlService, "riskScorerService", riskScorerService);
                ReflectionTestUtils.setField(riskControlService, "userRiskProfileService", userRiskProfileService);
        }

        @Test
        void testValidateOrder_HighRiskScore_ShouldThrowException() {
                ScoreResponse lowScoreResponse = ScoreResponse.newBuilder()
                                .setScore(500)
                                .setDecision("REJECT")
                                .build();
                when(riskScorerService.score(any(ScoreRequest.class))).thenReturn(lowScoreResponse);
                
                UserRiskProfile profile = UserRiskProfile.newBuilder()
                                .setUserId(1L)
                                .setAge(30)
                                .setExternalDebtRatio(0.1f)
                                .setGpsStability(0.9f)
                                .setDeviceSharedDegree(1)
                                .build();
                GetUserRiskProfileResponse profileResponse = GetUserRiskProfileResponse.newBuilder()
                                .setProfile(profile)
                                .build();
                when(userRiskProfileService.getUserRiskProfile(any(GetUserRiskProfileRequest.class))).thenReturn(profileResponse);

                assertThrows(RuntimeException.class,
                                () -> riskControlService.validateOrder(1L, new BigDecimal("100"), "USD"));
        }

        @Test
        void testValidateOrder_GoodRiskScore_ShouldPass() {
                ScoreResponse highScoreResponse = ScoreResponse.newBuilder()
                                .setScore(700)
                                .setDecision("APPROVE")
                                .build();
                when(riskScorerService.score(any(ScoreRequest.class))).thenReturn(highScoreResponse);
                
                UserRiskProfile profile = UserRiskProfile.newBuilder()
                                .setUserId(1L)
                                .setAge(30)
                                .setExternalDebtRatio(0.1f)
                                .setGpsStability(0.9f)
                                .setDeviceSharedDegree(1)
                                .build();
                GetUserRiskProfileResponse profileResponse = GetUserRiskProfileResponse.newBuilder()
                                .setProfile(profile)
                                .build();
                when(userRiskProfileService.getUserRiskProfile(any(GetUserRiskProfileRequest.class))).thenReturn(profileResponse);

                when(exchangeOrderRepository.getTotalCompletedAmountByUserIdAndDateRange(any(), any()))
                                .thenReturn(BigDecimal.ZERO);
                when(exchangeOrderRepository.getCompletedOrderCountByUserIdAndDateRange(any(), any()))
                                .thenReturn(0L);
                when(exchangeOrderRepository.findByUserIdAndStatus(any(), any()))
                                .thenReturn(java.util.Collections.emptyList());

                assertDoesNotThrow(() -> riskControlService.validateOrder(1L, new BigDecimal("100"), "USD"));
        }
}
