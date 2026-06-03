package com.metawebthree.cs.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.cs.application.TransferService;
import com.metawebthree.cs.domain.model.TransferLog;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/cs/transfer")
@Tag(name = "Transfer Controller", description = "会话转接接口")
public class TransferController {
    private final TransferService transferService;

    public TransferController(TransferService transferService) {
        this.transferService = transferService;
    }

    @Operation(summary = "转接会话")
    @PostMapping("/transfer")
    public ApiResponse<TransferLog> transfer(@Valid @RequestBody TransferRequest request) {
        TransferLog transferLog = transferService.transfer(
                request.sessionId,
                request.fromAgentId,
                request.toAgentId,
                request.reason);
        return ApiResponse.success(transferLog);
    }

    @Operation(summary = "转接历史")
    @GetMapping("/history")
    public ApiResponse<List<TransferLog>> history(@RequestParam String sessionId) {
        return ApiResponse.success(transferService.getTransferHistory(sessionId));
    }

    public static class TransferRequest {
        @NotBlank(message = "sessionId cannot be blank")
        public String sessionId;

        @NotNull(message = "fromAgentId cannot be null")
        public Long fromAgentId;

        @NotNull(message = "toAgentId cannot be null")
        public Long toAgentId;

        @NotBlank(message = "reason cannot be blank")
        public String reason;
    }
}