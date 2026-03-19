package com.metawebthree.commission.interfaces.web;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.commission.application.CommissionCommandService;
import com.metawebthree.commission.application.CommissionQueryService;
import com.metawebthree.commission.domain.CommissionAccount;
import com.metawebthree.commission.domain.CommissionRecord;
import com.metawebthree.commission.interfaces.web.dto.CommissionBalanceView;
import com.metawebthree.commission.interfaces.web.dto.CommissionBindRequest;
import com.metawebthree.commission.interfaces.web.dto.CommissionCalcRequest;
import com.metawebthree.commission.interfaces.web.dto.CommissionCancelRequest;
import com.metawebthree.commission.interfaces.web.dto.CommissionRecordView;
import com.metawebthree.commission.interfaces.web.dto.CommissionSettleRequest;

@RestController
@RequestMapping("/v1/commission")
public class CommissionController {
    private final CommissionCommandService commandService;
    private final CommissionQueryService queryService;

    public CommissionController(CommissionCommandService commandService, CommissionQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }

    @GetMapping("/health")
    public ApiResponse<String> health() {
        return ApiResponse.success("commission-service is running");
    }

    @PostMapping("/relations/bind")
    public ApiResponse<Void> bindRelation(@RequestBody CommissionBindRequest request) {
        commandService.bindRelation(request.getUserId(), request.getParentUserId());
        return ApiResponse.success();
    }

    @PostMapping("/calc")
    public ApiResponse<Void> calculate(@RequestBody CommissionCalcRequest request) {
        commandService.calculateForOrder(request.getOrderId(), request.getUserId(), request.getPayAmount(),
                request.getAvailableAt());
        return ApiResponse.success();
    }

    @PostMapping("/settle")
    public ApiResponse<Void> settle(@RequestBody CommissionSettleRequest request) {
        LocalDateTime executeBefore = request.getExecuteBefore();
        commandService.settleBefore(executeBefore);
        return ApiResponse.success();
    }

    @PostMapping("/cancel")
    public ApiResponse<Void> cancel(@RequestBody CommissionCancelRequest request) {
        commandService.cancelByOrderId(request.getOrderId());
        return ApiResponse.success();
    }

    @GetMapping("/balance")
    public ApiResponse<CommissionBalanceView> balance(@RequestParam Long userId) {
        CommissionAccount account = queryService.getAccount(userId);
        return ApiResponse.success(toBalanceView(userId, account));
    }

    @GetMapping("/ledger")
    public ApiResponse<List<CommissionRecordView>> ledger(@RequestParam Long userId,
            @RequestParam(defaultValue = "1") int page,
            @RequestParam(defaultValue = "20") int size,
            @RequestParam(required = false) String status) {
        List<CommissionRecord> records = queryService.listRecords(userId, status, page, size);
        return ApiResponse.success(toRecordViews(records));
    }

    private CommissionBalanceView toBalanceView(Long userId, CommissionAccount account) {
        CommissionBalanceView view = new CommissionBalanceView();
        view.setUserId(userId);
        view.setTotalAmount(valueOrZero(account == null ? null : account.getTotalAmount()));
        view.setAvailableAmount(valueOrZero(account == null ? null : account.getAvailableAmount()));
        view.setFrozenAmount(valueOrZero(account == null ? null : account.getFrozenAmount()));
        return view;
    }

    private List<CommissionRecordView> toRecordViews(List<CommissionRecord> records) {
        List<CommissionRecordView> views = new ArrayList<>();
        for (CommissionRecord record : records) {
            views.add(toRecordView(record));
        }
        return views;
    }

    private CommissionRecordView toRecordView(CommissionRecord record) {
        CommissionRecordView view = new CommissionRecordView();
        view.setId(record.getId());
        view.setOrderId(record.getOrderId());
        view.setFromUserId(record.getFromUserId());
        view.setLevel(record.getLevel());
        view.setAmount(record.getAmount());
        view.setStatus(record.getStatus());
        view.setAvailableAt(record.getAvailableAt());
        view.setCreatedAt(record.getCreatedAt());
        return view;
    }

    private BigDecimal valueOrZero(BigDecimal value) {
        return value == null ? BigDecimal.ZERO : value;
    }
}
