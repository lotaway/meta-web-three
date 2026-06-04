package com.metawebthree.recommendation.interfaces.graphql;

import com.apollographql.federation.graphqljava.Federation;
import com.metawebthree.recommendation.application.command.RecommendationCommandService;
import com.metawebthree.recommendation.application.query.RecommendationQueryService;
import com.metawebthree.recommendation.application.query.RecommendationQueryService.RecommendationMetrics;
import com.metawebthree.recommendation.domain.entity.Recommendation;
import com.metawebthree.recommendation.domain.entity.RecommendationResult;
import com.metawebthree.recommendation.domain.entity.RecommendationRule;
import com.metawebthree.recommendation.domain.entity.UserBehavior;
import com.metawebthree.recommendation.domain.repository.RecommendationRepository;
import com.metawebthree.recommendation.domain.repository.RecommendationRuleRepository;
import com.metawebthree.recommendation.infrastructure.persistence.entity.RecommendationResultDO;
import com.metawebthree.recommendation.infrastructure.persistence.entity.UserBehaviorDO;
import com.metawebthree.recommendation.infrastructure.persistence.mapper.RecommendationResultMapper;
import com.metawebthree.recommendation.infrastructure.persistence.mapper.UserBehaviorMapper;
import graphql.GraphQL;
import graphql.schema.GraphQLSchema;
import graphql.schema.idl.*;
import graphql.schema.DataFetchingEnvironment;
import graphql.TypeResolutionEnvironment;
import graphql.schema.GraphQLObjectType;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.ResourcePatternResolver;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@Configuration
public class RecommendationSubgraphConfig {

    private final RecommendationQueryService queryService;
    private final RecommendationCommandService commandService;
    private final RecommendationRepository recommendationRepository;
    private final RecommendationRuleRepository ruleRepository;
    private final RecommendationResultMapper recommendationResultMapper;
    private final UserBehaviorMapper userBehaviorMapper;
    private final ResourcePatternResolver resourceResolver;

    public RecommendationSubgraphConfig(
            RecommendationQueryService queryService,
            RecommendationCommandService commandService,
            RecommendationRepository recommendationRepository,
            RecommendationRuleRepository ruleRepository,
            RecommendationResultMapper recommendationResultMapper,
            UserBehaviorMapper userBehaviorMapper,
            ResourcePatternResolver resourceResolver) {
        this.queryService = queryService;
        this.commandService = commandService;
        this.recommendationRepository = recommendationRepository;
        this.ruleRepository = ruleRepository;
        this.recommendationResultMapper = recommendationResultMapper;
        this.userBehaviorMapper = userBehaviorMapper;
        this.resourceResolver = resourceResolver;
    }

    @Bean
    public GraphQL graphQL() throws IOException {
        TypeDefinitionRegistry typeRegistry = buildSchema();
        RuntimeWiring wiring = buildRuntimeWiring();
        GraphQLSchema schema = buildFederationSchema(typeRegistry, wiring);
        return GraphQL.newGraphQL(schema).build();
    }

    private TypeDefinitionRegistry buildSchema() throws IOException {
        Resource[] resources = resourceResolver.getResources("classpath:graphql/*.graphqls");
        StringBuilder sb = new StringBuilder();
        for (Resource r : resources) {
            sb.append(new String(r.getInputStream().readAllBytes(), StandardCharsets.UTF_8));
        }
        return new SchemaParser().parse(sb.toString());
    }

    private RuntimeWiring buildRuntimeWiring() {
        return RuntimeWiring.newRuntimeWiring()
                .type("Query", wiring -> wiring
                        .dataFetcher("recommendations", this::recommendationsDataFetcher)
                        .dataFetcher("recommendationsByScene", this::recommendationsBySceneDataFetcher)
                        .dataFetcher("recommendationsByAlgorithm", this::recommendationsByAlgorithmDataFetcher)
                        .dataFetcher("recommendation", this::recommendationDataFetcher)
                        .dataFetcher("recommendationMetrics", this::recommendationMetricsDataFetcher)
                        .dataFetcher("userBehaviorHistory", this::userBehaviorHistoryDataFetcher)
                        .dataFetcher("rulesByScene", this::rulesBySceneDataFetcher)
                )
                .type("Mutation", wiring -> wiring
                        .dataFetcher("generateRecommendation", this::generateRecommendationDataFetcher)
                        .dataFetcher("recordBehavior", this::recordBehaviorDataFetcher)
                        .dataFetcher("createRecommendationRule", this::createRecommendationRuleDataFetcher)
                        .dataFetcher("activateRecommendationRule", this::activateRecommendationRuleDataFetcher)
                        .dataFetcher("deleteRecommendationRule", this::deleteRecommendationRuleDataFetcher)
                        .dataFetcher("markRecommendationClicked", this::markRecommendationClickedDataFetcher)
                        .dataFetcher("markRecommendationPurchased", this::markRecommendationPurchasedDataFetcher)
                )
                .build();
    }

    private GraphQLSchema buildFederationSchema(TypeDefinitionRegistry registry, RuntimeWiring wiring) {
        return Federation.transform(registry, wiring)
                .fetchEntities(this::fetchEntity)
                .resolveEntityType(this::resolveEntityType)
                .build();
    }

    private Object fetchEntity(DataFetchingEnvironment env) {
        List<Map<String, Object>> representations = env.getArgument("representations");
        Map<String, Object> representation = representations.get(0);
        String typeName = (String) representation.get("__typename");
        Object idObj = representation.get("id");
        Long id = idObj instanceof Number ? ((Number) idObj).longValue() : Long.valueOf((String) idObj);

        if ("Recommendation".equals(typeName)) {
            return recommendationRepository.findById(id).orElse(null);
        }
        if ("RecommendationResult".equals(typeName)) {
            RecommendationResultDO doEntity = recommendationResultMapper.selectById(id);
            if (doEntity == null) return null;
            RecommendationResult r = new RecommendationResult();
            r.setId(doEntity.getId());
            r.setUserId(doEntity.getUserId());
            r.setProductId(doEntity.getProductId());
            r.setScore(doEntity.getScore());
            r.setAlgorithm(RecommendationResult.RecommendationAlgorithm.valueOf(doEntity.getAlgorithm()));
            r.setReason(doEntity.getReason());
            r.setPosition(doEntity.getPosition());
            r.setCreatedAt(doEntity.getCreatedAt());
            r.setExpiresAt(doEntity.getExpiresAt());
            r.setIsClicked(doEntity.getIsClicked() != null && doEntity.getIsClicked() == 1);
            r.setIsPurchased(doEntity.getIsPurchased() != null && doEntity.getIsPurchased() == 1);
            return r;
        }
        if ("RecommendationRule".equals(typeName)) {
            return ruleRepository.findById(id).orElse(null);
        }
        if ("UserBehavior".equals(typeName)) {
            UserBehaviorDO doEntity = userBehaviorMapper.selectById(id);
            if (doEntity == null) return null;
            UserBehavior b = new UserBehavior();
            b.setId(doEntity.getId());
            b.setUserId(doEntity.getUserId());
            b.setProductId(doEntity.getProductId());
            b.setBehaviorType(UserBehavior.BehaviorType.valueOf(doEntity.getBehaviorType()));
            b.setBehaviorValue(doEntity.getBehaviorValue());
            b.setTimestamp(doEntity.getTimestamp());
            b.setSessionId(doEntity.getSessionId());
            b.setSource(doEntity.getSource());
            return b;
        }
        return null;
    }

    private GraphQLObjectType resolveEntityType(TypeResolutionEnvironment env) {
        Object src = env.getObject();
        if (src instanceof Map) {
            String typeName = (String) ((Map<?, ?>) src).get("__typename");
            return env.getSchema().getObjectType(typeName);
        }
        return null;
    }

    private Object recommendationsDataFetcher(DataFetchingEnvironment env) {
        Long userId = Long.valueOf(env.getArgument("userId"));
        Integer limit = env.getArgument("limit");
        if (limit == null) limit = 10;
        return queryService.getRecommendations(userId, limit);
    }

    private List<Map<String, Object>> recommendationsBySceneDataFetcher(DataFetchingEnvironment env) {
        Long userId = Long.valueOf(env.getArgument("userId"));
        String scene = env.getArgument("scene");
        Integer limit = env.getArgument("limit");
        if (limit == null) limit = 10;
        List<Recommendation> recs = queryService.getUserRecommendationsByScene(userId, scene);
        return recs.stream().map(this::recommendationToResultMap).limit(limit).collect(Collectors.toList());
    }

    private List<RecommendationResult> recommendationsByAlgorithmDataFetcher(DataFetchingEnvironment env) {
        Long userId = Long.valueOf(env.getArgument("userId"));
        String algorithm = env.getArgument("algorithm");
        Integer limit = env.getArgument("limit");
        if (limit == null) limit = 10;
        RecommendationResult.RecommendationAlgorithm algo =
                RecommendationResult.RecommendationAlgorithm.valueOf(algorithm);
        return queryService.getRecommendationsByAlgorithm(userId, algo, limit);
    }

    private Object recommendationDataFetcher(DataFetchingEnvironment env) {
        String id = env.getArgument("id");
        return queryService.getRecommendationById(Long.valueOf(id)).orElse(null);
    }

    private RecommendationMetrics recommendationMetricsDataFetcher(DataFetchingEnvironment env) {
        Long userId = Long.valueOf(env.getArgument("userId"));
        return queryService.getRecommendationMetrics(userId);
    }

    private Object userBehaviorHistoryDataFetcher(DataFetchingEnvironment env) {
        Long userId = Long.valueOf(env.getArgument("userId"));
        Integer limit = env.getArgument("limit");
        if (limit == null) limit = 50;
        return queryService.getUserBehaviorHistory(userId, limit);
    }

    private Object rulesBySceneDataFetcher(DataFetchingEnvironment env) {
        String scene = env.getArgument("scene");
        return queryService.getRulesByScene(scene);
    }

    private Object generateRecommendationDataFetcher(DataFetchingEnvironment env) {
        Long userId = Long.valueOf(env.getArgument("userId"));
        String scene = env.getArgument("scene");
        String algorithm = env.getArgument("algorithm");
        Integer maxItems = env.getArgument("maxItems");
        if (maxItems == null) maxItems = 10;
        Recommendation.RecommendationAlgorithm algo =
                Recommendation.RecommendationAlgorithm.valueOf(algorithm);
        return commandService.generateRecommendation(userId, scene, algo, maxItems);
    }

    private Boolean recordBehaviorDataFetcher(DataFetchingEnvironment env) {
        Long userId = Long.valueOf(env.getArgument("userId"));
        String skuCode = env.getArgument("skuCode");
        String behaviorType = env.getArgument("behaviorType");
        commandService.recordBehavior(userId, skuCode, behaviorType);
        return true;
    }

    private Object createRecommendationRuleDataFetcher(DataFetchingEnvironment env) {
        String ruleName = env.getArgument("ruleName");
        String scene = env.getArgument("scene");
        String type = env.getArgument("type");
        RecommendationRule.RuleType ruleType = RecommendationRule.RuleType.valueOf(type);
        return commandService.createRule(ruleName, scene, ruleType);
    }

    private Boolean activateRecommendationRuleDataFetcher(DataFetchingEnvironment env) {
        String id = env.getArgument("id");
        commandService.activateRule(Long.valueOf(id));
        return true;
    }

    private Boolean deleteRecommendationRuleDataFetcher(DataFetchingEnvironment env) {
        String id = env.getArgument("id");
        commandService.deleteRule(Long.valueOf(id));
        return true;
    }

    private Boolean markRecommendationClickedDataFetcher(DataFetchingEnvironment env) {
        String id = env.getArgument("id");
        commandService.markRecommendationClicked(Long.valueOf(id));
        return true;
    }

    private Boolean markRecommendationPurchasedDataFetcher(DataFetchingEnvironment env) {
        String id = env.getArgument("id");
        commandService.markRecommendationPurchased(Long.valueOf(id));
        return true;
    }

    private Map<String, Object> recommendationToResultMap(Recommendation r) {
        Map<String, Object> map = new ConcurrentHashMap<>();
        map.put("id", r.getId().toString());
        map.put("userId", r.getUserId().toString());
        map.put("productId", "0");
        map.put("score", r.getScore() != null ? r.getScore().doubleValue() : null);
        map.put("algorithm", r.getAlgorithm() != null ? r.getAlgorithm().name() : null);
        map.put("reason", null);
        map.put("position", 0);
        map.put("isClicked", false);
        map.put("isPurchased", false);
        map.put("createdAt", r.getCreatedAt() != null ? r.getCreatedAt().toString() : null);
        map.put("expiresAt", r.getExpiresAt() != null ? r.getExpiresAt().toString() : null);
        return map;
    }
}
