package com.metawebthree.wallet.interfaces.controller;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.wallet.application.dto.ActivityDTO;
import com.metawebthree.wallet.application.dto.CreateActivityRequest;
import com.metawebthree.wallet.application.service.SolanaActivityService;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/solana/activities")
public class SolanaActivityController {

    private final SolanaActivityService activityService;

    public SolanaActivityController(SolanaActivityService activityService) {
        this.activityService = activityService;
    }

    @PostMapping
    public ApiResponse<ActivityDTO> createActivity(@RequestBody CreateActivityRequest request) {
        ActivityDTO result = activityService.createActivity(request);
        return ApiResponse.success(result);
    }

    @GetMapping
    public ApiResponse<List<ActivityDTO>> listActivities() {
        List<ActivityDTO> result = activityService.listActivities();
        return ApiResponse.success(result);
    }

    @GetMapping("/{activityAddress}")
    public ApiResponse<ActivityDTO> getActivity(@PathVariable String activityAddress) {
        ActivityDTO result = activityService.getActivity(activityAddress);
        return ApiResponse.success(result);
    }

    @PostMapping("/{activityAddress}/participate")
    public ApiResponse<ActivityDTO> participate(
            @PathVariable String activityAddress,
            @RequestBody Map<String, String> request) {
        ActivityDTO result = activityService.participate(activityAddress,
            request.get("participant"), request.get("authority"));
        return ApiResponse.success(result);
    }

    @PostMapping("/{activityAddress}/claim")
    public ApiResponse<String> claimReward(
            @PathVariable String activityAddress,
            @RequestBody Map<String, Object> request) {
        String winner = (String) request.get("winner");
        Integer rank = request.get("rank") != null ? ((Number) request.get("rank")).intValue() : null;
        String result = activityService.claimReward(activityAddress, winner, rank,
            (String) request.get("authority"));
        return ApiResponse.success(result);
    }
}
