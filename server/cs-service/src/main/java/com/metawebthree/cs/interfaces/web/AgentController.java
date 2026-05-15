package com.metawebthree.cs.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.cs.application.AgentService;
import com.metawebthree.cs.domain.model.Agent;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/cs/agent")
@Tag(name = "Agent Controller", description = "客服人员接口")
public class AgentController {
    private final AgentService agentService;

    public AgentController(AgentService agentService) {
        this.agentService = agentService;
    }

    @Operation(summary = "上線")
    @PostMapping("/online")
    public ApiResponse<Void> online(@RequestParam Long agentId) {
        agentService.goOnline(agentId);
        return ApiResponse.success();
    }

    @Operation(summary = "下线")
    @PostMapping("/offline")
    public ApiResponse<Void> offline(@RequestParam Long agentId) {
        agentService.goOffline(agentId);
        return ApiResponse.success();
    }

    @Operation(summary = "忙碌")
    @PostMapping("/busy")
    public ApiResponse<Void> busy(@RequestParam Long agentId) {
        agentService.setBusy(agentId);
        return ApiResponse.success();
    }

    @Operation(summary = "在线列表")
    @GetMapping("/online")
    public ApiResponse<List<Agent>> onlineList() {
        return ApiResponse.success(agentService.listOnline());
    }

    @Operation(summary = "根据ID查询")
    @GetMapping("/get")
    public ApiResponse<Agent> get(@RequestParam Long agentId) {
        return agentService.findById(agentId)
                .map(ApiResponse::success)
                .orElse(ApiResponse.error(
                        com.metawebthree.common.enums.ResponseStatus.NOT_FOUND));
    }
}
