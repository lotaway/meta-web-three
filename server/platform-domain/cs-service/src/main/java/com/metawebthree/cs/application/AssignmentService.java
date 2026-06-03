package com.metawebthree.cs.application;

import com.metawebthree.cs.domain.model.Agent;
import com.metawebthree.cs.domain.model.Conversation;
import com.metawebthree.cs.domain.repository.AgentRepository;
import com.metawebthree.cs.domain.repository.ConversationRepository;

import java.util.List;
import java.util.Optional;

import lombok.extern.slf4j.Slf4j;
import org.apache.dubbo.config.annotation.DubboReference;
import com.metawebthree.common.generated.rpc.*;

@Slf4j
public class AssignmentService {
    private static final Long GROUP_ID_PRODUCT = 1L;
    private static final Long GROUP_ID_ORDER = 2L;
    private static final Long GROUP_ID_AFTER_SALE = 3L;

    private final AgentRepository agentRepository;
    private final ConversationRepository conversationRepository;

    @DubboReference(check = false, lazy = true)
    private ProductService productService;

    @DubboReference(check = false, lazy = true)
    private OrderService orderService;

    public AssignmentService(AgentRepository agentRepository,
                              ConversationRepository conversationRepository) {
        this.agentRepository = agentRepository;
        this.conversationRepository = conversationRepository;
    }

    public Optional<Agent> findAvailableAgent(Conversation conversation) {
        Long previousAgentId = conversation.getAgentId();
        if (previousAgentId != null) {
            Optional<Agent> previousAgent = agentRepository.findById(previousAgentId);
            if (previousAgent.isPresent() && previousAgent.get().isAvailable()) {
                return previousAgent;
            }
        }
        Long groupId = findAgentGroupId(conversation);
        if (groupId != null) {
            List<Agent> available = agentRepository.findAvailableByGroupId(groupId);
            if (!available.isEmpty()) {
                return Optional.of(available.get(0));
            }
        }
        List<Agent> allOnline = agentRepository.findAvailableByGroupId(null);
        return allOnline.stream()
                .filter(Agent::isAvailable)
                .min((a, b) -> Integer.compare(a.getCurrentLoad(), b.getCurrentLoad()));
    }

    private Long findAgentGroupId(Conversation conversation) {
        Long productId = conversation.getProductId();
        Long orderId = conversation.getOrderId();

        if (productId != null) {
            try {
                GetProductDetailRequest request = GetProductDetailRequest.newBuilder()
                        .setProductId(productId)
                        .build();
                GetProductDetailResponse response = productService.getProductDetail(request);
                if (response != null && response.hasProduct()) {
                    ProductDetailProto product = response.getProduct();
                    long categoryId = product.getCategoryId();
                    if (isElectronicsCategory(categoryId)) {
                        return GROUP_ID_PRODUCT;
                    }
                    if (isFashionCategory(categoryId)) {
                        return 2L;
                    }
                    if (isFoodCategory(categoryId)) {
                        return 4L;
                    }
                }
            } catch (Exception e) {
                log.warn("failed to get product detail for productId {}: {}", productId, e.getMessage());
            }
        }

        if (orderId != null) {
            try {
                GetOrderByUserIdRequest request = GetOrderByUserIdRequest.newBuilder()
                        .setId(orderId)
                        .build();
                GetOrderByUserIdResponse response = orderService.getOrderByUserId(request);
                if (response != null && response.getOrdersCount() > 0) {
                    String orderType = response.getOrders(0).getOrderType();
                    if (isAfterSaleRelated(orderType)) {
                        return GROUP_ID_AFTER_SALE;
                    }
                    return GROUP_ID_ORDER;
                }
            } catch (Exception e) {
                log.warn("failed to get order detail for orderId {}: {}", orderId, e.getMessage());
            }
        }

        return null;
    }

    private boolean isElectronicsCategory(long categoryId) {
        return categoryId >= 1000 && categoryId < 2000;
    }

    private boolean isFashionCategory(long categoryId) {
        return categoryId >= 2000 && categoryId < 3000;
    }

    private boolean isFoodCategory(long categoryId) {
        return categoryId >= 3000 && categoryId < 4000;
    }

    private boolean isAfterSaleRelated(String orderType) {
        if (orderType == null) return false;
        String type = orderType.toLowerCase();
        return type.contains("refund") || type.contains("return") || type.contains("售后");
    }
}
