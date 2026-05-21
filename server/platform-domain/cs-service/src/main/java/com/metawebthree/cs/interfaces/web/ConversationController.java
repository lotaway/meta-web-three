package com.metawebthree.cs.interfaces.web;

import com.metawebthree.common.constants.HeaderConstants;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.cs.application.ConversationService;
import com.metawebthree.cs.domain.model.Conversation;
import com.metawebthree.cs.domain.model.enums.ChannelType;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/cs/conversation")
@Tag(name = "Conversation Controller", description = "客服会话接口")
public class ConversationController {
    private final ConversationService conversationService;

    public ConversationController(ConversationService conversationService) {
        this.conversationService = conversationService;
    }

    @Operation(summary = "创建会话")
    @PostMapping("/create")
    public ApiResponse<Conversation> create(
            @RequestHeader(HeaderConstants.USER_ID) Long userId,
            @RequestBody CreateRequest request) {
        Conversation conversation = conversationService.create(
                userId, request.channel, request.productId, request.orderId);
        return ApiResponse.success(conversation);
    }

    @Operation(summary = "关闭会话")
    @PostMapping("/close")
    public ApiResponse<Void> close(@RequestParam String sessionId) {
        conversationService.close(sessionId);
        return ApiResponse.success();
    }

    @Operation(summary = "打分评价")
    @PostMapping("/rate")
    public ApiResponse<Void> rate(@RequestParam String sessionId, @RequestParam Integer score) {
        conversationService.rate(sessionId, score);
        return ApiResponse.success();
    }

    @Operation(summary = "用户会话列表")
    @GetMapping("/my")
    public ApiResponse<List<Conversation>> myConversations(
            @RequestHeader(HeaderConstants.USER_ID) Long userId) {
        return ApiResponse.success(conversationService.listByCustomer(userId));
    }

    @Operation(summary = "客服会话列表")
    @GetMapping("/agent")
    public ApiResponse<List<Conversation>> agentConversations(@RequestParam Long agentId) {
        return ApiResponse.success(conversationService.listByAgent(agentId));
    }

    @Operation(summary = "排队列表")
    @GetMapping("/queuing")
    public ApiResponse<List<Conversation>> queuing() {
        return ApiResponse.success(conversationService.listQueuing());
    }

    public static class CreateRequest {
        public ChannelType channel;
        public Long productId;
        public Long orderId;
    }
}
