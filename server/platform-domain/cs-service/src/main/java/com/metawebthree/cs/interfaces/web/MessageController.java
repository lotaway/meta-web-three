package com.metawebthree.cs.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.cs.application.AiRoutingService;
import com.metawebthree.cs.application.MessageService;
import com.metawebthree.cs.domain.model.Message;
import com.metawebthree.cs.domain.model.enums.MessageType;
import com.metawebthree.cs.domain.model.enums.SenderType;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/cs/message")
@Tag(name = "Message Controller", description = "消息接口")
public class MessageController {
    private final MessageService messageService;
    private final AiRoutingService aiRoutingService;

    public MessageController(MessageService messageService, AiRoutingService aiRoutingService) {
        this.messageService = messageService;
        this.aiRoutingService = aiRoutingService;
    }

    @Operation(summary = "发送消息")
    @PostMapping("/send")
    public ApiResponse<Message> send(@RequestBody SendRequest request) {
        Message message = messageService.send(
                request.sessionId,
                SenderType.valueOf(request.senderType),
                request.senderId,
                MessageType.valueOf(request.msgType),
                request.content);
        return ApiResponse.success(message);
    }

    @Operation(summary = "AI 聊天")
    @PostMapping("/ai-chat")
    public ApiResponse<AiChatResponse> aiChat(@RequestBody AiChatRequest request) {
        String reply = aiRoutingService.processWithAi(
                request.sessionId,
                request.customerId,
                request.message);
        return ApiResponse.success(new AiChatResponse(reply));
    }

    @Operation(summary = "会话历史消息")
    @GetMapping("/list")
    public ApiResponse<List<Message>> list(@RequestParam String sessionId) {
        return ApiResponse.success(messageService.listBySession(sessionId));
    }

    public static class SendRequest {
        public String sessionId;
        public String senderType;
        public Long senderId;
        public String msgType;
        public String content;
    }

    public static class AiChatRequest {
        public String sessionId;
        public Long customerId;
        public String message;
    }

    public static class AiChatResponse {
        public String reply;

        public AiChatResponse() {}

        public AiChatResponse(String reply) {
            this.reply = reply;
        }
    }
}