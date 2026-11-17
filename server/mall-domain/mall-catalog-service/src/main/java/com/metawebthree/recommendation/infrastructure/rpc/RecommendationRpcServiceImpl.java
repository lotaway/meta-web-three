package com.metawebthree.recommendation.infrastructure.rpc;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;

import org.apache.dubbo.config.annotation.DubboService;

import com.metawebthree.common.generated.rpc.ActivateRuleRequest;
import com.metawebthree.common.generated.rpc.CreateRuleRequest;
import com.metawebthree.common.generated.rpc.CreateRuleResponse;
import com.metawebthree.common.generated.rpc.DeleteRuleRequest;
import com.metawebthree.common.generated.rpc.GenerateRecommendationRequest;
import com.metawebthree.common.generated.rpc.GenerateRecommendationResponse;
import com.metawebthree.common.generated.rpc.GenerateRecommendationsRequest;
import com.metawebthree.common.generated.rpc.GetBehaviorHistoryRequest;
import com.metawebthree.common.generated.rpc.GetBehaviorHistoryResponse;
import com.metawebthree.common.generated.rpc.GetMetricsRequest;
import com.metawebthree.common.generated.rpc.GetMetricsResponse;
import com.metawebthree.common.generated.rpc.GetRecommendationByIdRequest;
import com.metawebthree.common.generated.rpc.GetRecommendationByIdResponse;
import com.metawebthree.common.generated.rpc.GetRecommendationsByAlgorithmRequest;
import com.metawebthree.common.generated.rpc.GetRecommendationsByAlgorithmResponse;
import com.metawebthree.common.generated.rpc.GetRulesBySceneRequest;
import com.metawebthree.common.generated.rpc.GetRulesBySceneResponse;
import com.metawebthree.common.generated.rpc.GetUserRecommendationsBySceneRequest;
import com.metawebthree.common.generated.rpc.GetUserRecommendationsBySceneResponse;
import com.metawebthree.common.generated.rpc.GetUserRecommendationsRequest;
import com.metawebthree.common.generated.rpc.GetUserRecommendationsResponse;
import com.metawebthree.common.generated.rpc.MarkRecommendationRequest;
import com.metawebthree.common.generated.rpc.RecommendationMetricsView;
import com.metawebthree.common.generated.rpc.RecommendationResultView;
import com.metawebthree.common.generated.rpc.RecommendationRuleView;
import com.metawebthree.common.generated.rpc.RecommendationService;
import com.metawebthree.common.generated.rpc.RecommendationView;
import com.metawebthree.common.generated.rpc.RecommendationVoidResponse;
import com.metawebthree.common.generated.rpc.RecordBehaviorRequest;
import com.metawebthree.common.generated.rpc.RecommendedItemView;
import com.metawebthree.common.generated.rpc.UserBehaviorView;
import com.metawebthree.recommendation.application.command.RecommendationCommandService;
import com.metawebthree.recommendation.application.query.RecommendationQueryService;
import com.metawebthree.recommendation.application.query.RecommendationQueryService.RecommendationMetrics;
import com.metawebthree.recommendation.domain.entity.Recommendation;
import com.metawebthree.recommendation.domain.entity.RecommendationResult;
import com.metawebthree.recommendation.domain.entity.RecommendationRule;
import com.metawebthree.recommendation.domain.entity.UserBehavior;

/**
 * Dubbo provider for the gateway's recommendation client. Delegates to the
 * same application services used by the REST controller so behaviour matches
 * the former HTTP endpoint exactly.
 */
@DubboService
public class RecommendationRpcServiceImpl implements RecommendationService {

    private final RecommendationCommandService commandService;
    private final RecommendationQueryService queryService;

    public RecommendationRpcServiceImpl(RecommendationCommandService commandService,
                                         RecommendationQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }

    @Override
    public GenerateRecommendationResponse generateRecommendation(GenerateRecommendationRequest request) {
        Recommendation algorithm = commandService.generateRecommendation(
                request.getUserId(), request.getScene(),
                Recommendation.RecommendationAlgorithm.valueOf(request.getAlgorithm().toUpperCase()),
                request.getMaxItems());
        return GenerateRecommendationResponse.newBuilder()
                .setRecommendation(toView(algorithm)).build();
    }

    @Override
    public GetRecommendationByIdResponse getRecommendationById(GetRecommendationByIdRequest request) {
        GetRecommendationByIdResponse.Builder response = GetRecommendationByIdResponse.newBuilder();
        Optional<Recommendation> found = queryService.getRecommendationById(request.getId());
        found.ifPresent(value -> response.setRecommendation(toView(value)));
        return response.build();
    }

    @Override
    public GetUserRecommendationsResponse getUserRecommendations(GetUserRecommendationsRequest request) {
        GetUserRecommendationsResponse.Builder response = GetUserRecommendationsResponse.newBuilder();
        queryService.getUserRecommendations(request.getUserId())
                .forEach(value -> response.addRecommendations(toView(value)));
        return response.build();
    }

    @Override
    public GetUserRecommendationsBySceneResponse getUserRecommendationsByScene(
            GetUserRecommendationsBySceneRequest request) {
        GetUserRecommendationsBySceneResponse.Builder response =
                GetUserRecommendationsBySceneResponse.newBuilder();
        queryService.getUserRecommendationsByScene(request.getUserId(), request.getScene())
                .forEach(value -> response.addRecommendations(toView(value)));
        return response.build();
    }

    @Override
    public GetRecommendationsByAlgorithmResponse getRecommendationsByAlgorithm(
            GetRecommendationsByAlgorithmRequest request) {
        GetRecommendationsByAlgorithmResponse.Builder response =
                GetRecommendationsByAlgorithmResponse.newBuilder();
        queryService.getRecommendationsByAlgorithm(request.getUserId(),
                        RecommendationResult.RecommendationAlgorithm.valueOf(
                                request.getAlgorithm().toUpperCase()), request.getLimit())
                .forEach(value -> response.addResults(toResultView(value)));
        return response.build();
    }

    @Override
    public GetRecommendationsByAlgorithmResponse generateRecommendations(GenerateRecommendationsRequest request) {
        GetRecommendationsByAlgorithmResponse.Builder response =
                GetRecommendationsByAlgorithmResponse.newBuilder();
        commandService.generateRecommendationsByAlgorithm(request.getUserId(), null, request.getLimit())
                .forEach(value -> response.addResults(toResultView(value)));
        return response.build();
    }

    @Override
    public RecommendationVoidResponse recordBehavior(RecordBehaviorRequest request) {
        commandService.recordBehavior(request.getUserId(), request.getSkuCode(), request.getBehaviorType());
        return voidResponse();
    }

    @Override
    public RecommendationVoidResponse markAsClicked(MarkRecommendationRequest request) {
        commandService.markRecommendationClicked(request.getRecommendationId());
        return voidResponse();
    }

    @Override
    public RecommendationVoidResponse markAsPurchased(MarkRecommendationRequest request) {
        commandService.markRecommendationPurchased(request.getRecommendationId());
        return voidResponse();
    }

    @Override
    public GetBehaviorHistoryResponse getBehaviorHistory(GetBehaviorHistoryRequest request) {
        GetBehaviorHistoryResponse.Builder response = GetBehaviorHistoryResponse.newBuilder();
        queryService.getUserBehaviorHistory(request.getUserId(), request.getLimit())
                .forEach(value -> response.addBehaviors(toBehaviorView(value)));
        return response.build();
    }

    @Override
    public GetMetricsResponse getMetrics(GetMetricsRequest request) {
        RecommendationMetrics metrics = queryService.getRecommendationMetrics(request.getUserId());
        return GetMetricsResponse.newBuilder().setMetrics(toMetricsView(metrics)).build();
    }

    @Override
    public GetRulesBySceneResponse getRulesByScene(GetRulesBySceneRequest request) {
        GetRulesBySceneResponse.Builder response = GetRulesBySceneResponse.newBuilder();
        queryService.getRulesByScene(request.getScene())
                .forEach(value -> response.addRules(toRuleView(value)));
        return response.build();
    }

    @Override
    public CreateRuleResponse createRule(CreateRuleRequest request) {
        RecommendationRule rule = commandService.createRule(request.getRuleName(), request.getScene(),
                RecommendationRule.RuleType.valueOf(request.getType().toUpperCase()));
        return CreateRuleResponse.newBuilder().setRule(toRuleView(rule)).build();
    }

    @Override
    public RecommendationVoidResponse activateRule(ActivateRuleRequest request) {
        commandService.activateRule(request.getId());
        return voidResponse();
    }

    @Override
    public RecommendationVoidResponse deleteRule(DeleteRuleRequest request) {
        commandService.deleteRule(request.getId());
        return voidResponse();
    }

    // Async methods delegate to their synchronous counterparts.
    @Override
    public CompletableFuture<GenerateRecommendationResponse> generateRecommendationAsync(GenerateRecommendationRequest r) {
        return CompletableFuture.completedFuture(generateRecommendation(r));
    }
    @Override
    public CompletableFuture<GetRecommendationByIdResponse> getRecommendationByIdAsync(GetRecommendationByIdRequest r) {
        return CompletableFuture.completedFuture(getRecommendationById(r));
    }
    @Override
    public CompletableFuture<GetUserRecommendationsResponse> getUserRecommendationsAsync(GetUserRecommendationsRequest r) {
        return CompletableFuture.completedFuture(getUserRecommendations(r));
    }
    @Override
    public CompletableFuture<GetUserRecommendationsBySceneResponse> getUserRecommendationsBySceneAsync(GetUserRecommendationsBySceneRequest r) {
        return CompletableFuture.completedFuture(getUserRecommendationsByScene(r));
    }
    @Override
    public CompletableFuture<GetRecommendationsByAlgorithmResponse> getRecommendationsByAlgorithmAsync(GetRecommendationsByAlgorithmRequest r) {
        return CompletableFuture.completedFuture(getRecommendationsByAlgorithm(r));
    }
    @Override
    public CompletableFuture<GetRecommendationsByAlgorithmResponse> generateRecommendationsAsync(GenerateRecommendationsRequest r) {
        return CompletableFuture.completedFuture(generateRecommendations(r));
    }
    @Override
    public CompletableFuture<RecommendationVoidResponse> recordBehaviorAsync(RecordBehaviorRequest r) {
        return CompletableFuture.completedFuture(recordBehavior(r));
    }
    @Override
    public CompletableFuture<RecommendationVoidResponse> markAsClickedAsync(MarkRecommendationRequest r) {
        return CompletableFuture.completedFuture(markAsClicked(r));
    }
    @Override
    public CompletableFuture<RecommendationVoidResponse> markAsPurchasedAsync(MarkRecommendationRequest r) {
        return CompletableFuture.completedFuture(markAsPurchased(r));
    }
    @Override
    public CompletableFuture<GetBehaviorHistoryResponse> getBehaviorHistoryAsync(GetBehaviorHistoryRequest r) {
        return CompletableFuture.completedFuture(getBehaviorHistory(r));
    }
    @Override
    public CompletableFuture<GetMetricsResponse> getMetricsAsync(GetMetricsRequest r) {
        return CompletableFuture.completedFuture(getMetrics(r));
    }
    @Override
    public CompletableFuture<GetRulesBySceneResponse> getRulesBySceneAsync(GetRulesBySceneRequest r) {
        return CompletableFuture.completedFuture(getRulesByScene(r));
    }
    @Override
    public CompletableFuture<CreateRuleResponse> createRuleAsync(CreateRuleRequest r) {
        return CompletableFuture.completedFuture(createRule(r));
    }
    @Override
    public CompletableFuture<RecommendationVoidResponse> activateRuleAsync(ActivateRuleRequest r) {
        return CompletableFuture.completedFuture(activateRule(r));
    }
    @Override
    public CompletableFuture<RecommendationVoidResponse> deleteRuleAsync(DeleteRuleRequest r) {
        return CompletableFuture.completedFuture(deleteRule(r));
    }

    private RecommendationVoidResponse voidResponse() {
        return RecommendationVoidResponse.newBuilder().setSuccess(true).build();
    }

    private RecommendationView toView(Recommendation value) {
        RecommendationView.Builder view = RecommendationView.newBuilder()
                .setId(value.getId() == null ? 0 : value.getId())
                .setUserId(value.getUserId() == null ? 0 : value.getUserId())
                .setScene(nullToEmpty(value.getScene()))
                .setAlgorithm(value.getAlgorithm() == null ? "" : value.getAlgorithm().name())
                .setScore(decimal(value.getScore()))
                .setStatus(value.getStatus() == null ? "" : value.getStatus().name())
                .setClickCount(number(value.getClickCount()))
                .setConversionCount(number(value.getConversionCount()))
                .setImpressionCount(number(value.getImpressionCount()))
                .setCreatedAt(time(value.getCreatedAt()))
                .setExpiresAt(time(value.getExpiresAt()));
        if (value.getItems() != null) {
            value.getItems().forEach(item -> view.addItems(RecommendedItemView.newBuilder()
                    .setSkuCode(nullToEmpty(item.getSkuCode()))
                    .setSkuName(nullToEmpty(item.getSkuName()))
                    .setScore(decimal(item.getScore()))
                    .setRank(number(item.getRank()))
                    .setReason(nullToEmpty(item.getReason())).build()));
        }
        return view.build();
    }

    private RecommendationResultView toResultView(RecommendationResult value) {
        return RecommendationResultView.newBuilder()
                .setId(value.getId() == null ? 0 : value.getId())
                .setUserId(value.getUserId() == null ? 0 : value.getUserId())
                .setProductId(value.getProductId() == null ? 0 : value.getProductId())
                .setScore(value.getScore() == null ? 0 : value.getScore())
                .setAlgorithm(value.getAlgorithm() == null ? "" : value.getAlgorithm().name())
                .setReason(nullToEmpty(value.getReason()))
                .setPosition(number(value.getPosition()))
                .setIsClicked(Boolean.TRUE.equals(value.getIsClicked()))
                .setIsPurchased(Boolean.TRUE.equals(value.getIsPurchased()))
                .setCreatedAt(time(value.getCreatedAt()))
                .setExpiresAt(time(value.getExpiresAt()))
                .build();
    }

    private RecommendationRuleView toRuleView(RecommendationRule value) {
        return RecommendationRuleView.newBuilder()
                .setId(value.getId() == null ? 0 : value.getId())
                .setRuleName(nullToEmpty(value.getRuleName()))
                .setScene(nullToEmpty(value.getScene()))
                .setType(value.getType() == null ? "" : value.getType().name())
                .setStatus(value.getStatus() == null ? "" : value.getStatus().name())
                .setPriority(number(value.getPriority()))
                .setMaxItems(number(value.getMaxItems()))
                .setConditions(nullToEmpty(value.getConditions()))
                .setExclusions(nullToEmpty(value.getExclusions()))
                .setBoostFactor(decimal(value.getBoostFactor()))
                .setCreatedAt(time(value.getCreatedAt()))
                .setUpdatedAt(time(value.getUpdatedAt()))
                .build();
    }

    private UserBehaviorView toBehaviorView(UserBehavior value) {
        return UserBehaviorView.newBuilder()
                .setId(value.getId() == null ? 0 : value.getId())
                .setUserId(value.getUserId() == null ? 0 : value.getUserId())
                .setProductId(value.getProductId() == null ? 0 : value.getProductId())
                .setBehaviorType(value.getBehaviorType() == null ? "" : value.getBehaviorType().name())
                .setBehaviorValue(value.getBehaviorValue() == null ? 0 : value.getBehaviorValue())
                .setTimestamp(time(value.getTimestamp()))
                .setSessionId(nullToEmpty(value.getSessionId()))
                .setSource(nullToEmpty(value.getSource()))
                .build();
    }

    private RecommendationMetricsView toMetricsView(RecommendationMetrics value) {
        return RecommendationMetricsView.newBuilder()
                .setTotalRecommendations(value.getTotalRecommendations())
                .setClickedCount(value.getClickedCount())
                .setPurchasedCount(value.getPurchasedCount())
                .setClickThroughRate(value.getClickThroughRate())
                .setConversionRate(value.getConversionRate())
                .build();
    }

    private String nullToEmpty(String value) { return value == null ? "" : value; }
    private int number(Integer value) { return value == null ? 0 : value; }
    private double decimal(BigDecimal value) { return value == null ? 0 : value.doubleValue(); }
    private String time(LocalDateTime value) { return value == null ? "" : value.toString(); }
}
