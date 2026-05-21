package com.metawebthree.commission.infrastructure.rpc;

import java.math.BigDecimal;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;

import org.apache.dubbo.config.annotation.DubboService;

import com.metawebthree.commission.application.CommissionCommandService;
import com.metawebthree.common.generated.rpc.BindRequest;
import com.metawebthree.common.generated.rpc.BindResponse;
import com.metawebthree.common.generated.rpc.CalculateRequest;
import com.metawebthree.common.generated.rpc.CalculateResponse;
import com.metawebthree.common.generated.rpc.CancelRequest;
import com.metawebthree.common.generated.rpc.CancelResponse;
import com.metawebthree.common.generated.rpc.CommissionService;

import java.util.concurrent.CompletableFuture;

@DubboService
public class CommissionRPCServiceImpl implements CommissionService {
    private final CommissionCommandService commandService;

    public CommissionRPCServiceImpl(CommissionCommandService commandService) {
        this.commandService = commandService;
    }

    @Override
    public CalculateResponse calculate(CalculateRequest request) {
        BigDecimal payAmount = new BigDecimal(request.getPayAmount());
        LocalDateTime availableAt = LocalDateTime.ofInstant(
                Instant.ofEpochMilli(request.getAvailableAt()), ZoneId.systemDefault());
        
        commandService.calculateForOrder(request.getOrderId(), request.getUserId(), payAmount, availableAt);
        
        return CalculateResponse.newBuilder().setSuccess(true).build();
    }

    @Override
    public CompletableFuture<CalculateResponse> calculateAsync(CalculateRequest request) {
        return CompletableFuture.completedFuture(calculate(request));
    }

    @Override
    public CancelResponse cancel(CancelRequest request) {
        commandService.cancelByOrderId(request.getOrderId());
        return CancelResponse.newBuilder().setSuccess(true).build();
    }

    @Override
    public CompletableFuture<CancelResponse> cancelAsync(CancelRequest request) {
        return CompletableFuture.completedFuture(cancel(request));
    }

    @Override
    public BindResponse bind(BindRequest request) {
        commandService.bindRelation(request.getUserId(), request.getParentUserId());
        return BindResponse.newBuilder().setSuccess(true).build();
    }

    @Override
    public CompletableFuture<BindResponse> bindAsync(BindRequest request) {
        return CompletableFuture.completedFuture(bind(request));
    }
}
