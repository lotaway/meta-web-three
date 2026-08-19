package com.metawebthree.message.infrastructure.rpc;

import java.util.concurrent.CompletableFuture;

import org.apache.dubbo.config.annotation.DubboService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.metawebthree.common.generated.rpc.platform.MessageService;
import com.metawebthree.common.generated.rpc.platform.SendEmailRequest;
import com.metawebthree.common.generated.rpc.platform.SendEmailResponse;
import com.metawebthree.email.EmailSendService;

@DubboService
public class MessageRPCServiceImpl implements MessageService {

    private static final Logger logger = LoggerFactory.getLogger(MessageRPCServiceImpl.class);

    private final EmailSendService emailSendService;

    public MessageRPCServiceImpl(EmailSendService emailSendService) {
        this.emailSendService = emailSendService;
    }

    @Override
    public SendEmailResponse sendEmail(SendEmailRequest request) {
        boolean success = emailSendService.send(request.getTo(), request.getTitle(), request.getContent());
        if (!success) {
            logger.warn("Email delivery failed via Dubbo: to={}", request.getTo());
        }
        return SendEmailResponse.newBuilder().setSuccess(success).build();
    }

    @Override
    public CompletableFuture<SendEmailResponse> sendEmailAsync(SendEmailRequest request) {
        return CompletableFuture.completedFuture(sendEmail(request));
    }
}