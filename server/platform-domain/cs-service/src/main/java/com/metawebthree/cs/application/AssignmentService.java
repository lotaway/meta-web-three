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
        // Try to determine agent group based on product category or order type
        Long productId = conversation.getProductId();
        Long orderId = conversation.getOrderId();

        if (productId != null) {
            try {
                GetProductDetailRequest request = GetProductDetailRequest.newBuilder()
                        .setProductId(productId)
                        .build();
                GetProductDetailResponse response = productService.getProductDetail(request);
                if (response != null && response.hasProduct()) {
                    // Currently returns null - group assignment can be enhanced later
                    // with product category-based routing logic
                    log.debug("Found product for conversation {}, category routing not implemented",
                            conversation.getId());
                }
            } catch (Exception e) {
                log.warn("Failed to get product detail for productId {}: {}", productId, e.getMessage());
            }
        }

        if (orderId != null) {
            try {
                GetOrderByUserIdRequest request = GetOrderByUserIdRequest.newBuilder()
                        .setId(orderId)
                        .build();
                GetOrderByUserIdResponse response = orderService.getOrderByUserId(request);
                if (response != null && response.getOrdersCount() > 0) {
                    // Could determine group based on order type
                    log.debug("Found order for conversation {}, type: {}",
                            conversation.getId(), response.getOrders(0).getOrderType());
                }
            } catch (Exception e) {
                log.warn("Failed to get order detail for orderId {}: {}", orderId, e.getMessage());
            }
        }

        // Return null to allow any available agent to handle the conversation
        // Group assignment can be enhanced with more business logic
        return null;
    }
}
