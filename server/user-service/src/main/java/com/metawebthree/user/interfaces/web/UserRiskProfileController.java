package com.metawebthree.user.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.dto.UserRiskProfileUpdateDTO;
import com.metawebthree.common.rpc.UserRiskProfileService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RestController
@RequestMapping("/user/risk-profile")
@RequiredArgsConstructor
public class UserRiskProfileController {

    private final UserRiskProfileService userRiskProfileService;

    @PostMapping("/update")
    public ApiResponse<Void> updateRiskProfile(@RequestBody UserRiskProfileUpdateDTO updateDTO) {
        log.info("Received request to update risk profile for user: {}", updateDTO.getUserId());
        try {
            userRiskProfileService.updateRiskProfile(updateDTO);
            return ApiResponse.success();
        } catch (Exception e) {
            log.error("Failed to update risk profile", e);
            return ApiResponse.error(com.metawebthree.common.enums.ResponseStatus.SYSTEM_ERROR, e.getMessage());
        }
    }
}
